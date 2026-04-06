"""
DeepSeek-V3 + 法条检索增强（RAG）
- 使用 shibing624/text2vec-base-chinese 进行向量检索
- 每次调用 API 前，先从 Chroma 向量库检索最相近的 3 条法条
- 将法条信息加入到 Prompt 中
- 结果保存到 model_output/zero_shot/deepseek_v3_with_retriever
"""

import os
import json
import logging
import http.client
import glob
import chromadb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import threading
import time

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
API_KEYS = [
    "sk-aZ4KJUKxqQGI0kvJBf87Aa2eE53a487e81A8D26f89CfD486",
    "sk-DJ29rXEPynvLeo1oF3B760C896Cd426cA3B08452AfF9904e",
    "sk-0ZUB8rg4mn837rP97139C01d1cF34fD6A275A651B219Ba36",
    "sk-TBnjH5MJFpVkRVnjF3E352E065F34e4fA4B395Dd16EcBe90",
    "sk-WDc1G15qcXMGqKe562A43c14Ba934807B3Bd30F8A74a3729",
    "sk-KOSjINQK6Hq6nYKq687045C77aAa4cB394Fc8e337886E49b",
    "sk-1uVFVa56Vf1UJI5EF6BfAeD331D443B49b0d1cDaB5D2E8F2",
    "sk-flmDwsSl2Nm4RK3e677cC84f904a4dE2Be0d4a11D584FcC7",
    "sk-yIP2MhrdtnzzJWU2D82bC5F313144b5cA48aCaD2126aB5A1",
    "sk-yFXgUqnTSVsr0Xhc1595245c35E347A5B7BaCf7e72Bc3cE9"
]

API_HOST = 'dpapi.cn'
MODEL_NAME = 'deepseek-v3'

DATA_DIR = './data'
OUT_DIR = './model_output/zero_shot/deepseek_v3_with_retriever'
LOG_FILE = 'deepseek_v3_with_retriever_running.log'

# 向量库配置
CHROMA_PATH = "./chroma_db"
COLLECTION = "china_law"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
TOP_K_LAWS = 3  # 检索最相近的 3 条法条

MAX_RETRY = 3
NUM_WORKERS = 10
# ════════════════════════════════════════════════════════════

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── API Key 线程安全分配 ──────────────────────────────────────
thread_local = threading.local()
_key_lock = threading.Lock()
_key_index = 0


def get_api_key():
    global _key_index
    if not hasattr(thread_local, "api_key"):
        with _key_lock:
            thread_local.api_key = API_KEYS[_key_index % len(API_KEYS)]
            _key_index += 1
    return thread_local.api_key


# ── 法条检索器 ───────────────────────────────────────────────
class LawRetriever:
    """
    使用 Chroma + Sentence Transformers 进行法条向量检索
    """

    def __init__(self):
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = client.get_collection(COLLECTION)
            self.available = True
            print(f"✅ 向量库已加载，共 {self.collection.count()} 条记录")
            logging.info(f"Law retriever initialized. Total records: {self.collection.count()}")
        except Exception as e:
            print(f"⚠️ 向量库加载失败: {e}，将继续运行但不使用检索功能")
            logging.warning(f"Law retriever init failed: {e}")
            self.available = False

    def search(self, query: str, top_k: int = 3, law_filter: str = None) -> list:
        """
        检索相关法条
        返回: [{'law': ..., 'article': ..., 'text': ..., 'score': ...}, ...]
        """
        if not self.available:
            return []

        try:
            query_vector = self.model.encode(query).tolist()
            where = {"law": law_filter} if law_filter else None

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            hits = []
            for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
            ):
                hits.append({
                    "score": round(1 - dist, 4),
                    "law": meta.get("law", ""),
                    "article": meta.get("article", ""),
                    "chapter": meta.get("chapter", ""),
                    "text": meta.get("text", ""),
                    "chunk": doc,
                })
            return hits
        except Exception as e:
            logging.warning(f"Search failed: {e}")
            return []


# ── 全局检索器实例（单例）──────────────────────────────────────
_retriever = None
_retriever_lock = threading.Lock()


def get_retriever():
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = LawRetriever()
    return _retriever


# ── 数据加载 ─────────────────────────────────────────────────
def load_data(f_path):
    data = []
    with open(f_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not data:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data


# ── Prompt 构建（增强版：加入检索的法条）────────────────────
SUFFIX_MAP = {
    '5_1': '摘要:',
    '5_2': '裁判分析过程:',
    '5_3': '翻译结果:',
}


def build_prompt_with_retrieval(item, task_name, retrieved_laws: list) -> str:
    """
    构建增强 Prompt：原始内容 + 检索的法条信息
    """
    base_prompt = item.get('instruction', '') + item.get('input', '')

    # ── 构建法条参考部分 ──────────────────────────────────────
    if retrieved_laws:
        laws_text = "\n\n【相关法条参考】"
        for i, law in enumerate(retrieved_laws, 1):
            laws_text += f"\n{i}. {law['law']} {law['article']}"
            laws_text += f"\n   {law['text']}"
            if law['chapter']:
                laws_text += f"\n   (章节: {law['chapter']})"
            laws_text += f"\n   [相似度: {law['score']:.2%}]"
        base_prompt = base_prompt + laws_text

    suffix = SUFFIX_MAP.get(task_name, '答案:')
    return base_prompt + '\n' + suffix


def build_prompt(item, task_name):
    """原始 Prompt 构建（不使用检索）"""
    suffix = SUFFIX_MAP.get(task_name, '答案:')
    return item.get('instruction', '') + item.get('input', '') + '\n' + suffix


# ── API 调用 ──────────────────────────────────────────────
def call_deepseek(prompt: str) -> str:
    api_key = get_api_key()
    conn = http.client.HTTPSConnection(API_HOST)
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    })
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    raw = res.read().decode('utf-8')
    data = json.loads(raw)

    if 'choices' not in data:
        raise ValueError(f"API returned unexpected response: {data}")

    return data['choices'][0]['message']['content']


# ── 单条处理（带检索）────────────────────────────────────────
def process_item_with_retrieval(item, task_name):
    """
    处理单条数据：
    1. 用 input 检索法条
    2. 构建增强 Prompt
    3. 调用 API
    """
    retriever = get_retriever()
    answer = item.get('answer', '')
    item_input = item.get('input', '')

    # ── 检索相关法条 ──────────────────────────────────────────
    retrieved_laws = []
    if retriever.available and item_input:
        try:
            retrieved_laws = retriever.search(item_input, top_k=TOP_K_LAWS)
        except Exception as e:
            logging.warning(f"[{task_name}] Retrieval failed: {e}")
            retrieved_laws = []

    # ── 构建增强 Prompt ────────────────────────────────────
    prompt = build_prompt_with_retrieval(item, task_name, retrieved_laws)

    # ── API 调用（重试）───────────────────────────────────────
    for attempt in range(MAX_RETRY):
        try:
            response = call_deepseek(prompt)
            return {
                "input": item_input,
                "output": response,
                "answer": answer,
                "retrieved_laws": [
                    {
                        "law": law['law'],
                        "article": law['article'],
                        "score": law['score']
                    }
                    for law in retrieved_laws
                ]
            }
        except Exception as e:
            logging.warning(f"[{task_name}] attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRY - 1:
                return {
                    "input": item_input,
                    "output": "FAILED",
                    "answer": answer,
                    "retrieved_laws": []
                }
            time.sleep(1)


# ── 单文件处理 ───────────────────────────────────────────────
def run_on_file(f_path):
    task_name = os.path.basename(f_path).split('.')[0]
    data = load_data(f_path)

    print(f"\n🚀 Running {task_name} (RAG) | {len(data)} samples")
    logging.info(f"Start running deepseek_v3 with retrieval on task {task_name}")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_item_with_retrieval, item, task_name): item
            for item in data
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=task_name):
            results.append(future.result())

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"deepseek_v3_with_retriever_{task_name}.jsonl")

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    failed = sum(1 for r in results if r['output'] == 'FAILED')
    print(f"✅ Saved → {out_path}  (failed: {failed}/{len(results)})")
    if failed:
        logging.warning(f"[{task_name}] {failed}/{len(results)} items failed")


# ── 入口 ─────────────────────────────────────────────────────
def main():
    # 初始化检索器
    print("Initializing law retriever...")
    get_retriever()
    print()

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))

    if not all_files:
        print("❌ No data found in", DATA_DIR)
        return

    print(f"Found {len(all_files)} files")
    for f in all_files:
        run_on_file(f)

    print("\n🎉 ALL DONE (with retrieval)")


if __name__ == "__main__":
    main()
