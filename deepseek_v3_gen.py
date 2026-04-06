import os
import json
import logging
import http.client
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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

API_HOST    = 'dpapi.cn'
MODEL_NAME  = 'deepseek-v3'

DATA_DIR    = './data'
OUT_DIR     = './model_output/zero_shot/deepseek_v3'
LOG_FILE    = 'deepseek_v3_running.log'

MAX_RETRY   = 3
NUM_WORKERS = 10
# ════════════════════════════════════════════════════════════

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── API Key 线程安全分配 ──────────────────────────────────────
thread_local = threading.local()
_key_lock    = threading.Lock()
_key_index   = 0

def get_api_key():
    global _key_index
    if not hasattr(thread_local, "api_key"):
        with _key_lock:
            thread_local.api_key = API_KEYS[_key_index % len(API_KEYS)]
            _key_index += 1
    return thread_local.api_key


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


# ── Prompt 构建（与 LexEval 原始逻辑对齐，按 task 加后缀）────
SUFFIX_MAP = {
    '5_1': '摘要:',
    '5_2': '裁判分析过程:',
    '5_3': '翻译结果:',
}

def build_prompt(item, task_name):
    suffix = SUFFIX_MAP.get(task_name, '答案:')
    return item.get('instruction', '') + item.get('input', '') + '\n' + suffix


# ── API 调用（修复：加 Bearer 前缀）──────────────────────────
def call_deepseek(prompt: str) -> str:
    api_key = get_api_key()

    conn = http.client.HTTPSConnection(API_HOST)
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    })
    headers = {
        'Authorization': f'Bearer {api_key}',   # ← 修复点
        'Content-Type': 'application/json'
    }

    conn.request("POST", "/v1/chat/completions", payload, headers)
    res  = conn.getresponse()
    raw  = res.read().decode('utf-8')
    data = json.loads(raw)

    if 'choices' not in data:
        raise ValueError(f"API returned unexpected response: {data}")

    return data['choices'][0]['message']['content']


# ── 单条处理 ─────────────────────────────────────────────────
def process_item(item, task_name):
    prompt = build_prompt(item, task_name)
    answer = item.get('answer', '')

    for attempt in range(MAX_RETRY):
        try:
            response = call_deepseek(prompt)
            return {
                "input":  item.get("input", ""),
                "output": response,
                "answer": answer
            }
        except Exception as e:
            logging.warning(f"[{task_name}] attempt {attempt+1} failed: {e}")
            if attempt == MAX_RETRY - 1:
                return {
                    "input":  item.get("input", ""),
                    "output": "FAILED",
                    "answer": answer
                }
            time.sleep(1)


# ── 单文件处理 ───────────────────────────────────────────────
def run_on_file(f_path):
    task_name = os.path.basename(f_path).split('.')[0]   # e.g. "1_1"
    data      = load_data(f_path)

    print(f"\n🚀 Running {task_name} | {len(data)} samples")
    logging.info(f"Start running deepseek_v3 on task {task_name}")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_item, item, task_name): item for item in data}
        for future in tqdm(as_completed(futures), total=len(futures), desc=task_name):
            results.append(future.result())

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"deepseek_v3_{task_name}.jsonl")

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    failed = sum(1 for r in results if r['output'] == 'FAILED')
    print(f"✅ Saved → {out_path}  (failed: {failed}/{len(results)})")
    if failed:
        logging.warning(f"[{task_name}] {failed}/{len(results)} items failed")


# ── 入口 ─────────────────────────────────────────────────────
def main():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))

    if not all_files:
        print("❌ No data found in", DATA_DIR)
        return

    print(f"Found {len(all_files)} files")
    for f in all_files:
        run_on_file(f)

    print("\n🎉 ALL DONE")


if __name__ == "__main__":
    main()