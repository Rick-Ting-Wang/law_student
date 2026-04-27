"""
指定从哪里开始测试
"""
import os
import json
import logging
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ════════════════════════════════════════════════════════════
# CONFIG（固定）
# ════════════════════════════════════════════════════════════
MODEL_NAME = "Qwen/Qwen-7B-Chat"

DATA_DIR = './data'
OUT_DIR = './model_output/qwen_7B_fp16'
LOG_FILE = 'qwen_chat.log'

MAX_RETRY = 3
NUM_WORKERS = 1   # ⚠️ 必须=1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 直接写死：从这里开始
START_TASK = "4_2"

# ════════════════════════════════════════════════════════════

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── 全局模型（单例）────────────────────────
_model = None
_tokenizer = None
_model_lock = threading.Lock()


def get_model_and_tokenizer():
    global _model, _tokenizer

    if _model is None:
        with _model_lock:
            if _model is None:
                print("📥 Loading Qwen-7B-Chat...")
                logging.info("Loading model")

                _tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True
                )

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


# ── 数据加载 ───────────────────────────────
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


# ── Prompt 构建 ───────────────────────────
SUFFIX_MAP = {
    '5_1': '摘要:',
    '5_2': '裁判分析过程:',
    '5_3': '翻译结果:',
}


def build_prompt(item, task_name):
    suffix = SUFFIX_MAP.get(task_name, '答案:')
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')

    return f"{instruction}\n{input_text}\n{suffix}"


# ── 推理 ────────────────────────────────
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


# ── 单条处理 ─────────────────────────────
def process_item(item, task_name):
    prompt = build_prompt(item, task_name)
    answer = item.get('answer', '')

    for attempt in range(MAX_RETRY):
        try:
            response = call_qwen_local(prompt)

            return {
                "input": item.get("input", ""),
                "output": response,
                "answer": answer
            }

        except Exception as e:
            logging.warning(f"[{task_name}] attempt {attempt+1} failed: {e}")

            if attempt == MAX_RETRY - 1:
                return {
                    "input": item.get("input", ""),
                    "output": "FAILED",
                    "answer": answer
                }


# ── 单文件处理 ───────────────────────────
def run_on_file(f_path):
    task_name = os.path.basename(f_path).split('.')[0]
    data = load_data(f_path)

    print(f"\n🚀 Running {task_name} | {len(data)} samples")
    logging.info(f"Start task {task_name}")

    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_item, item, task_name): item
            for item in data
        }

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


# ── 主函数（核心改动）────────────────────
def main():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))

    if not all_files:
        print("❌ No data found in", DATA_DIR)
        return

    print(f"Found {len(all_files)} files")

    start_running = False

    for f in all_files:
        task_name = os.path.basename(f).split('.')[0]

        if not start_running:
            if task_name == START_TASK:
                start_running = True
                print(f"\n🎯 Start from {START_TASK}")
            else:
                print(f"⏭️ Skip {task_name}")
                continue

        run_on_file(f)

    print("\n🎉 ALL DONE")


if __name__ == "__main__":
    main()