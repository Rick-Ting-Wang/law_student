import os
import json
import logging
import glob
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
MODEL_NAME = "Qwen/Qwen-7B-Chat"

DATA_DIR = './data'
OUT_DIR = './model_output/zero_shot/qwen_7B_fp16_with_retriever'
LOG_FILE = 'qwen_chat.log'
LAW_DATA_DIR = './law_structured'      # 法条 JSON 文件目录
CHROMA_PATH = "./chroma_db"            # 向量库路径
COLLECTION_NAME = "china_law"
VECTOR_MODEL = "shibing624/text2vec-base-chinese"

MAX_RETRY = 3
NUM_WORKERS = 1   # 单线程稳定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 新增：从哪个文件开始继续运行（跳过之前的文件）
START_FROM = "3_1"   # 对应文件 2_2.json

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── 全局变量（模型 + 法律检索器）────────────────────────────
_model = None
_tokenizer = None
_model_lock = threading.Lock()

_law_retriever = None
_law_index = {}          # 精确匹配索引: key = "法律名称|条文号" -> 法条详情
_law_index_lock = threading.Lock()


# ════════════════════════════════════════════════════════════
# 法律检索模块（精确匹配 + 向量检索）
# ════════════════════════════════════════════════════════════
def load_law_structured():
    """加载 ./law_structured 下所有 JSON 文件，构建精确匹配索引"""
    global _law_index
    if _law_index:
        return

    with _law_index_lock:
        if _law_index:
            return

        law_files = glob.glob(os.path.join(LAW_DATA_DIR, '*.json'))
        if not law_files:
            logging.warning("未找到法律条文文件，法律检索功能将不可用")
            return

        for fpath in law_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]  # 兼容单条
                    for item in data:
                        law = item.get('law', '').strip()
                        article = item.get('article', '').strip()
                        if law and article:
                            key = f"{law}|{article}"
                            _law_index[key] = item
            except Exception as e:
                logging.error(f"加载法律文件失败 {fpath}: {e}")

        logging.info(f"精确匹配索引构建完成，共 {len(_law_index)} 条")


def extract_law_article(text: str):
    """
    从文本中提取法律名称和条文号
    返回列表，每个元素为 (law_name, article_num_str)
    示例：'刑法第二百三十八条' -> ('刑法', '第二百三十八条')
          '民法典第675条' -> ('民法典', '675')
          '中华人民共和国刑法第二百三十八条' -> ('中华人民共和国刑法', '第二百三十八条')
    """
    patterns = [
        # 匹配 "XXX法第N条" 或 "XXX法第N条之一"
        r'([\u4e00-\u9fa5]{2,}?(?:法|条例|规定))第([零一二三四五六七八九十百千万\d]+条(?:之[一二三四五])?)',
        # 匹配 "XXX法第N条" 无“第”字
        r'([\u4e00-\u9fa5]{2,}?(?:法|条例|规定))([零一二三四五六七八九十百千万\d]+条(?:之[一二三四五])?)',
        # 匹配 "刑法第二百三十八条" 形式
        r'([\u4e00-\u9fa5]{2,}?(?:法|条例|规定))([零一二三四五六七八九十百千万]+条)',
    ]
    results = []
    for pat in patterns:
        matches = re.findall(pat, text)
        for law, article in matches:
            results.append((law.strip(), article.strip()))
    # 去重
    return list(set(results))


def exact_match_law(law_name: str, article: str):
    """精确匹配法律条文，返回法条内容（dict）或 None"""
    if not _law_index:
        load_law_structured()
    # 标准化法律名称（去除空格，支持简写）
    law_name = law_name.strip()
    article = article.strip()
    # 直接匹配
    key = f"{law_name}|{article}"
    if key in _law_index:
        return _law_index[key]
    # 尝试忽略“中华人民共和国”前缀
    if law_name.startswith("中华人民共和国"):
        short = law_name[5:]
        key2 = f"{short}|{article}"
        if key2 in _law_index:
            return _law_index[key2]
    # 尝试将数字转换为中文（例如 "675" -> "六百七十五"）-- 简单处理，不完美但常见
    # 实际上条文号通常是中文数字或阿拉伯数字，这里留给向量检索兜底
    return None


def get_law_retriever():
    """懒加载向量检索器"""
    global _law_retriever
    if _law_retriever is None:
        try:
            # 检查依赖
            import chromadb
            from sentence_transformers import SentenceTransformer
            # 初始化
            model = SentenceTransformer(VECTOR_MODEL)
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_collection(COLLECTION_NAME)
            _law_retriever = (model, collection)
            logging.info("向量检索器加载成功")
        except Exception as e:
            logging.error(f"向量检索器加载失败: {e}")
            _law_retriever = False
    return _law_retriever


def vector_search_law(query: str, top_k: int = 2):
    """使用向量检索返回相似法条列表"""
    ret = get_law_retriever()
    if not ret or ret is False:
        return []
    model, collection = ret
    try:
        query_vec = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            hits.append({
                "score": 1 - dist,
                "law": meta["law"],
                "article": meta["article"],
                "chapter": meta.get("chapter", ""),
                "text": meta["text"],
                "chunk": doc,
            })
        return hits
    except Exception as e:
        logging.error(f"向量检索失败: {e}")
        return []


def retrieve_law_context(input_text: str) -> str:
    """
    综合精确匹配和向量检索，返回适合放入 Prompt 的法律条文上下文。
    返回字符串，如果没有找到则返回空字符串。
    """
    context_parts = []
    # 1. 精确匹配
    extracted = extract_law_article(input_text)
    for law_name, article in extracted:
        match = exact_match_law(law_name, article)
        if match:
            chunk = match.get('chunk', '')
            if chunk:
                context_parts.append(f"【精确匹配】{chunk}")
            else:
                context_parts.append(f"【精确匹配】{match.get('law')} {match.get('article')}: {match.get('text', '')}")
    # 2. 如果精确匹配没有结果，则进行向量检索（取 top2）
    if not context_parts:
        vector_hits = vector_search_law(input_text, top_k=2)
        for hit in vector_hits:
            context_parts.append(f"【相似法条】(相似度{hit['score']:.2%}) {hit['law']} {hit['article']}: {hit['text']}")
    # 格式化返回
    if context_parts:
        return "\n" + "\n".join(context_parts) + "\n"
    return ""


# ════════════════════════════════════════════════════════════
# 模型加载与 Prompt 构建
# ════════════════════════════════════════════════════════════
def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        with _model_lock:
            if _model is None:
                print("📥 Loading Qwen-7B-Chat...")
                logging.info("Loading model")
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).eval()
                print("✅ Model loaded")
                logging.info("Model loaded")
    return _model, _tokenizer


SUFFIX_MAP = {
    '5_1': '摘要:',
    '5_2': '裁判分析过程:',
    '5_3': '翻译结果:',
}


def is_multiple_choice_task(filename: str, instruction: str = "") -> bool:
    """判断是否为选择题任务（用于添加不定项选择提示）"""
    # 根据文件名模式：数字_数字.json（如 1_1, 2_3, 3_6）
    if re.match(r'^\d+_\d+\.json$', filename):
        return True
    # 或者根据 instruction 关键词
    if instruction and "选择题" in instruction:
        return True
    return False


def build_prompt(item, task_name, law_context: str = ""):
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')

    # 选择题任务：明确标注不定项选择
    if is_multiple_choice_task(task_name + ".json", instruction):
        if "选择题" in instruction:
            # 在原有指令中加入不定项说明
            instruction = instruction.replace("选择题", "不定项选择题")
        else:
            instruction = instruction + "（本题为不定项选择题）"

    prompt = f"{instruction}\n{input_text}\n"

    if law_context:
        prompt += f"\n【参考法律条文】{law_context}\n"

    prompt += "答案:"
    return prompt


def load_data(f_path):
    data = []
    with open(f_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except:
                    pass
    if not data:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data


def call_qwen_local(prompt: str) -> str:
    model, tokenizer = get_model_and_tokenizer()
    try:
        response, _ = model.chat(
            tokenizer,
            prompt,
            history=None,
            do_sample=False
        )
        return response.strip()
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise


def process_item(item, task_name):
    input_text = item.get('input', '')
    # 检索法律上下文（仅当输入文本较长或包含法律关键词时，可自行决定是否检索）
    law_context = retrieve_law_context(input_text) if input_text else ""

    prompt = build_prompt(item, task_name, law_context)
    answer = item.get('answer', '')

    for attempt in range(MAX_RETRY):
        try:
            response = call_qwen_local(prompt)
            return {
                "input": item.get("input", ""),
                "output": response,
                "answer": answer,
                "law_context": law_context.strip()   # 记录检索到的法条，便于调试
            }
        except Exception as e:
            logging.warning(f"[{task_name}] attempt {attempt+1} failed: {e}")
            if attempt == MAX_RETRY - 1:
                return {
                    "input": item.get("input", ""),
                    "output": "FAILED",
                    "answer": answer,
                    "law_context": ""
                }


def run_on_file(f_path):
    task_name = os.path.basename(f_path).split('.')[0]
    data = load_data(f_path)

    print(f"\n🚀 Running {task_name} | {len(data)} samples")
    logging.info(f"Start task {task_name}")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_item, item, task_name): item for item in data}
        for future in tqdm(as_completed(futures), total=len(futures), desc=task_name):
            results.append(future.result())

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"qwen_{task_name}.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    failed = sum(1 for r in results if r['output'] == 'FAILED')
    print(f"✅ Saved → {out_path}  (failed: {failed}/{len(results)})")
    if failed:
        logging.warning(f"{task_name}: {failed} failed")


def parse_filename(filename: str):
    """解析类似 '1_2.json' 的文件名，返回 (part1, part2) 整数元组，用于排序和比较"""
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    else:
        # 非标准格式，返回一个很大的值使其排在最后
        return (float('inf'), float('inf'))


def main():
    # 预先加载法律索引
    load_law_structured()
    # 可选：预加载向量检索器（但会在第一次使用时加载）

    # 获取所有 JSON 文件并按照数字顺序排序
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')), key=parse_filename)

    if not all_files:
        print("❌ No data found in", DATA_DIR)
        return

    # 解析起始文件的比较键
    start_key = parse_filename(START_FROM + ".json")

    # 过滤：只保留 >= start_key 的文件
    filtered_files = [f for f in all_files if parse_filename(os.path.basename(f)) >= start_key]

    if not filtered_files:
        print(f"❌ No files found from {START_FROM} onward")
        return

    print(f"Found {len(all_files)} total files, starting from {START_FROM} → {len(filtered_files)} files to process")

    for f in filtered_files:
        run_on_file(f)

    print("\n🎉 ALL DONE")


if __name__ == "__main__":
    main()