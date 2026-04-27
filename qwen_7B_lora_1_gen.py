import os
import json
import logging
import glob
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
BASE_MODEL_NAME = "Qwen/Qwen-7B-Chat"
LORA_PATH = "./output_qwen_lora/lora"

DATA_DIR = './data'
OUT_DIR = './model_output/zero_shot/qwen_7B_lora'
LOG_FILE = 'qwen_lora_chat.log'

MAX_RETRY = 3
TARGET_TASK = "3_6"  # ⭐ 单选题
MAX_NEW_TOKENS = 100

# ════════════════════════════════════════════════════════════

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

_model = None
_tokenizer = None


def get_model_and_tokenizer():
    global _model, _tokenizer

    if _model is None:
        print("📥 Loading Base Model...")
        logging.info("Loading model")

        _tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True,
            use_fast=False
        )
        _tokenizer.pad_token_id = _tokenizer.eod_id

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval()

        print("📥 Loading LoRA weights...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            LORA_PATH,
            torch_dtype=torch.float16,
        )

        print("🔧 Merging LoRA weights...")
        _model = peft_model.merge_and_unload()
        _model.eval()

        print("✅ Model ready!")
        logging.info("Model ready")

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


# ── Prompt构建 ────────────────────────────
# ⭐ 单选题 (3_6) 专用 Prompt
TASK_CONFIG = {
    '3_6': {
        'task_type': '单选题',
        'suffix': '答案:',
        'system_prompt': '你是一个智能助手。对于单选题，请选择最合适的选项。'
    },
    '5_1': {
        'task_type': '摘要',
        'suffix': '摘要:',
        'system_prompt': '你是一个智能助手'
    },
    '5_2': {
        'task_type': '裁判分析',
        'suffix': '裁判分析过程:',
        'system_prompt': '你是一个智能助手'
    },
    '5_3': {
        'task_type': '翻译',
        'suffix': '翻译结果:',
        'system_prompt': '你是一个智能助手'
    },
}


def build_prompt(item, task_name):
    config = TASK_CONFIG.get(task_name, {})
    system_prompt = config.get('system_prompt', '你是一个智能助手')
    suffix = config.get('suffix', '答案:')

    instruction = item.get('instruction', '')
    input_text = item.get('input', '')

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{suffix}"
    )
    return prompt


# ── 推理 ─────────────────────────────────
def call_qwen_local(prompt: str) -> str:
    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eod_id,
            pad_token_id=tokenizer.eod_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


# ── 单条处理 ──────────────────────────────
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
            logging.warning(f"[{task_name}] attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRY - 1:
                return {
                    "input": item.get("input", ""),
                    "output": "FAILED",
                    "answer": answer
                }


# ── 单文件处理 ────────────────────────────
def run_on_file(f_path, task_name):
    data = load_data(f_path)
    task_type = TASK_CONFIG.get(task_name, {}).get('task_type', 'Unknown')

    print(f"\n{'=' * 60}")
    print(f"🚀 Running Task: {task_name} ({task_type})")
    print(f"📊 Samples: {len(data)}")
    print(f"{'=' * 60}")

    logging.info(f"Start task {task_name} - Type: {task_type}")

    results = []

    for item in tqdm(data, desc=f"{task_name} [{task_type}]"):
        results.append(process_item(item, task_name))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"qwen_{task_name}.jsonl")

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    failed = sum(1 for r in results if r['output'] == 'FAILED')
    success_rate = (len(results) - failed) / len(results) * 100

    print(f"\n✅ Saved → {out_path}")
    print(f"📈 Success Rate: {success_rate:.1f}% ({len(results) - failed}/{len(results)})")

    if failed:
        logging.warning(f"{task_name}: {failed} failed")


# ── 主函数 ────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("⭐ SINGLE CHOICE QUESTION (3_6) INFERENCE PIPELINE")
    print("=" * 60)

    print("🔧 Pre-loading model...")
    get_model_and_tokenizer()

    # 直接运行 3_6 单选题任务
    task_file = os.path.join(DATA_DIR, f'{TARGET_TASK}.json')

    if not os.path.exists(task_file):
        print(f"❌ Task file not found: {task_file}")
        logging.error(f"Task file not found: {task_file}")
        return

    run_on_file(task_file, TARGET_TASK)

    print("\n" + "=" * 60)
    print("🎉 SINGLE CHOICE QUESTION (3_6) INFERENCE COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()