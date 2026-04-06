#重新运行 1_2、2_1、3_2 任务：

"""
DeepSeek-V3 + 法条检索增强（RAG）- 补丁版（10线程）
- 运行 1_2、2_1、3_2 任务
- 10 个 API Key，每个 Key 绑定 1 个线程，总共 10 个线程
- 其余逻辑与主脚本完全一致
"""

import os
import json
import logging
import http.client
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
LOG_FILE = 'deepseek_v3_with_retriever_patch_running.log'

# 向量库配置
CHROMA_PATH = "./chroma_db"
COLLECTION = "china_law"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
TOP_K_LAWS = 3

# ── 补丁专属配置 ─────────────────────────────────────────────
TASKS_TO_RUN = ['1_2', '2_1', '3_2']  # ✅ 修改为新任务
MAX_RETRY = 3
NUM_WORKERS = len(API_KEYS)  # 10 个线程，每个 Key 一个线程
# ════════════════════════════════════════════════════════════

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── API Key 线程安全分配（每个线程绑定一个 Key）──────────────
thread_local = threading.local()
_key_lock = threading.Lock()
_key_index = 0


def get_api_key():
    """
    每个线程首次调用时分配一个 Key，之后复用该 Key
    10 个线程 × 10 个 Key = 1:1 对应
    """
    if not hasattr(thread_local, "api_key"):
        global _key_index
        with _key_lock:
            thread_local.api_key = API_KEYS[_key_index % len(API_KEYS)]
            _key_index += 1
    return thread_local.api_key


# ── 法条检索器 ────────────────────────────────────────────────
class LawRetriever:
    def __init__(self):
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = client.get_collection(COLLECTION)
            self.available = True
            print(f"✅ Law retriever ready: {self.collection.count()} records")
            logging.info(f"Law retriever initialized. Total: {self.collection.count()}")
        except Exception as e:
            print(f"⚠️ Law retriever init failed: {e}")
            logging.warning(f"Law retriever init failed: {e}")
            self.available = False

    def search(self, query: str, top_k: int = 3) -> list:
        if not self.available or not query:
            return []
        try:
            query_vector = self.model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
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
                })
            return hits
        except Exception as e:
            logging.warning(f"Search failed: {e}")
            return []


# ── 全局检索器单例 ─────────────────────────────────────────────
_retriever = None
_retriever_lock = threading.Lock()


def get_retriever():
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = LawRetriever()
    return _retriever


# ── 数据加载 ───────────────────────────────────────────────────
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


# ── Prompt 构建 ────────────────────────────────────────────────
SUFFIX_MAP = {
    '1_2': '答案:',
    '2_1': '答案:',
    '3_2': '答案:',
}


def build_prompt_with_retrieval(item, task_name, retrieved_laws: list) -> str:
    base_prompt = item.get('instruction', '') + item.get('input', '')

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


# ── API 调用 ────────────────────────────────────────────────
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
        raise ValueError(f"Unexpected API response: {data}")

    return data['choices'][0]['message']['content']


# ── 单条处理 ───────────────────────────────────────────────
def process_item(item, task_name):
    retriever = get_retriever()
    answer = item.get('answer', '')
    item_input = item.get('input', '')

    # 检索法条
    retrieved_laws = []
    if retriever.available and item_input:
        try:
            retrieved_laws = retriever.search(item_input, top_k=TOP_K_LAWS)
        except Exception as e:
            logging.warning(f"[{task_name}] Retrieval failed: {e}")

    prompt = build_prompt_with_retrieval(item, task_name, retrieved_laws)

    # API 调用（重试）
    for attempt in range(MAX_RETRY):
        try:
            response = call_deepseek(prompt)

            # ── 打印 response ──────────────────────────────────
            print(f"\n[{task_name}] ✅ Response:")
            print(f"  Input: {item_input[:100]}...")
            print(f"  Output: {response[:200]}...")
            print()
            # ────────────────────────────────────────────────────

            return {
                "input": item_input,
                "output": response,
                "answer": answer,
                "retrieved_laws": [
                    {"law": l['law'], "article": l['article'], "score": l['score']}
                    for l in retrieved_laws
                ]
            }
        except Exception as e:
            print(f"[{task_name}] ❌ Attempt {attempt + 1} failed: {e}")
            logging.warning(f"[{task_name}] attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRY - 1:
                return {
                    "input": item_input,
                    "output": "FAILED",
                    "answer": answer,
                    "retrieved_laws": []
                }
            time.sleep(2)


# ── 单文件处理 ─────────────────────────────────────────────
def run_on_file(f_path):
    task_name = os.path.basename(f_path).split('.')[0]
    data = load_data(f_path)

    print(f"\n🚀 Running {task_name} ({NUM_WORKERS} workers) | {len(data)} samples")
    logging.info(f"Start task {task_name} | {len(data)} samples | workers={NUM_WORKERS}")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_item, item, task_name): item
            for item in data
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=task_name):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"[{task_name}] Future raised exception: {e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"deepseek_v3_with_retriever_{task_name}.jsonl")

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    failed = sum(1 for r in results if r['output'] == 'FAILED')
    print(f"✅ Saved → {out_path}  (failed: {failed}/{len(results)})")
    if failed:
        logging.warning(f"[{task_name}] {failed}/{len(results)} items failed")


# ── 入口 ────────────────────────────────────────────────────
def main():
    print(f"[PATCH] Running tasks: {TASKS_TO_RUN}")
    print(f"[PATCH] Workers: {NUM_WORKERS} (1 thread per API Key)\n")

    # 初始化检索器
    print("Initializing law retriever...")
    get_retriever()
    print()

    for task_name in TASKS_TO_RUN:
        f_path = os.path.join(DATA_DIR, f'{task_name}.json')
        if not os.path.exists(f_path):
            print(f"[SKIP] File not found: {f_path}")
            logging.warning(f"File not found: {f_path}")
            continue
        run_on_file(f_path)

    print("\n🎉 ALL DONE (patch - 10 workers)")
    logging.info("Patch script completed.")


if __name__ == "__main__":
    main()

