import os
import json
import logging
import http.client
import glob
import time
import threading

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
LOG_FILE = 'deepseek_v3_with_retriever_fix.log'
MAX_RETRY = 50

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── API Key 轮询 ──────────────────────────────────────────────
_key_index = 0
_key_lock = threading.Lock()


def get_api_key():
    global _key_index
    with _key_lock:
        key = API_KEYS[_key_index % len(API_KEYS)]
        _key_index += 1
    return key


# ── Prompt 构建 ────────────────────────────────────────────────
SUFFIX_MAP = {
    '5_1': '摘要:',
    '5_2': '裁判分析过程:',
    '5_3': '翻译结果:',
}


def build_prompt(item, task_name):
    suffix = SUFFIX_MAP.get(task_name, '答案:')
    return item.get('instruction', '') + item.get('input', '') + '\n' + suffix


# ── 原始数据加载 ───────────────────────────────────────────────
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


# ── API 调用 ──────────────────────────────────────────────────
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


# ── 重试直到成功 ───────────────────────────────────────────────
def retry_call(prompt: str, task_name: str, item_input: str) -> str:
    for attempt in range(MAX_RETRY):
        try:
            response = call_deepseek(prompt)
            logging.info(f"[{task_name}] Fix succeeded on attempt {attempt + 1} | input[:50]={item_input[:50]}")
            return response
        except Exception as e:
            logging.warning(f"[{task_name}] Fix attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** min(attempt, 5))  # 指数退避，最多等 32s

    raise RuntimeError(f"[{task_name}] All {MAX_RETRY} retries exhausted for input: {item_input[:80]}")


# ════════════════════════════════════════════════════════════
# 核心 Fix 逻辑
# ════════════════════════════════════════════════════════════
def fix_jsonl(out_path: str, data_path: str, task_name: str):
    """修复单个 JSONL 文件中的所有 FAILED 项"""

    # ── Step 1: 读取输出 JSONL ────────────────────────────────
    records = []
    with open(out_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    failed_indices = [i for i, r in enumerate(records) if r.get('output') == 'FAILED']

    if not failed_indices:
        print(f"  [OK]   {task_name}: no FAILED entries found.")
        return

    print(f"\n  [FIX]  {task_name}: {len(failed_indices)} FAILED item(s) at indices {failed_indices}")
    logging.info(f"[{task_name}] Fixing {len(failed_indices)} failed items: indices={failed_indices}")

    # ── Step 2: 建立 input -> 原始数据 映射 ────────────────────
    raw_data = load_data(data_path)
    input_to_raw = {}
    for item in raw_data:
        key = item.get('input', '')
        if key and key not in input_to_raw:
            input_to_raw[key] = item

    # ── Step 3: 逐条重新请求 ──────────────────────────────────
    fixed = 0
    for idx in failed_indices:
        failed_record = records[idx]
        item_input = failed_record.get('input', '')

        raw_item = input_to_raw.get(item_input)
        if raw_item is None:
            print(f"    [WARN] Cannot find original data for index {idx}, input[:60]={item_input[:60]}")
            logging.warning(f"[{task_name}] Original item not found for index {idx}")
            continue

        prompt = build_prompt(raw_item, task_name)
        print(f"    [QUERY] index {idx} | input[:60]={item_input[:60]!r}")

        try:
            new_output = retry_call(prompt, task_name, item_input)
            records[idx]['output'] = new_output
            print(f"    [OK]   index {idx} fixed. output[:80]={new_output[:80]!r}")
            fixed += 1
        except RuntimeError as e:
            print(f"    [FAIL] {e}")
            logging.error(str(e))

    # ── Step 4: 原地回写 JSONL ────────────────────────────────
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # ── Step 5: 验证结果 ──────────────────────────────────────
    still_failed = sum(1 for r in records if r.get('output') == 'FAILED')
    if still_failed == 0:
        print(f"  [DONE] {task_name}: all {fixed} items fixed successfully ✅")
        logging.info(f"[{task_name}] All {fixed} failed items fixed.")
    else:
        print(f"  [WARN] {task_name}: {fixed} fixed, {still_failed} still FAILED ⚠️")
        logging.warning(f"[{task_name}] {fixed} fixed, {still_failed} still failed.")


# ════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════
def main():
    # 扫描输出目录中所有 JSONL
    out_files = sorted(glob.glob(os.path.join(OUT_DIR, 'deepseek_v3_with_retriever_*.jsonl')))
    if not out_files:
        print(f"❌ No output JSONL files found in {OUT_DIR}")
        return

    print(f"🔍 Found {len(out_files)} output file(s), scanning for FAILED entries...\n")

    fixed_count = 0
    for out_path in out_files:
        basename = os.path.basename(out_path)
        # deepseek_v3_with_retriever_6_2.jsonl -> 6_2
        task_name = basename.replace('deepseek_v3_with_retriever_', '').replace('.jsonl', '')

        # 对应原始数据文件
        data_path = os.path.join(DATA_DIR, f"{task_name}.json")
        if not os.path.exists(data_path):
            print(f"  [SKIP] {task_name}: source data file not found ({data_path})")
            continue

        # 快速预检：是否有 FAILED
        has_failed = False
        with open(out_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '"FAILED"' in line:
                    has_failed = True
                    break

        if not has_failed:
            print(f"  [OK]   {task_name}: no FAILED entries, skipping.")
            continue

        fix_jsonl(out_path, data_path, task_name)
        fixed_count += 1

    print(f"\n{'=' * 60}")
    print(f"✅ All done. Processed {fixed_count} file(s).")
    print(f"{'=' * 60}")
    logging.info(f"Fix script completed. Processed {fixed_count} file(s).")


if __name__ == "__main__":
    main()
