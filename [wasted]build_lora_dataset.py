"""
[wasted]build_lora_dataset.py
=====================
功能：
  1. 扫描 lawtxtversion/ 文件夹，去掉文件名时间戳，提取法律名称
  2. 用 Qwen 模型（本地 or API）回答 "xxx是什么"
  3. 把 (问题, 模型回答, 真实法条) 构造成 LoRA 微调数据集（alpaca 格式）
  4. 输出 dataset.json 供 LLaMA-Factory / 官方 finetune.py 直接使用

使用方式：
  pip install transformers torch tqdm
  python [wasted]build_lora_dataset.py \
    --law_dir ./lawtxtversion \
    --output ./lora_dataset.json \
    --model_path Qwen/Qwen-7B-Chat   # 或本地路径
"""



import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────
# 1. 解析文件名，去掉时间戳（8位数字结尾）
# ──────────────────────────────────────────────
def parse_law_name(filename: str) -> str:
    """
    '艾滋病防治条例_20190302.txt'  →  '艾滋病防治条例'
    '保税区海关监管办法_20110108.txt' → '保税区海关监管办法'
    """
    stem = Path(filename).stem          # 去掉 .txt
    name = re.sub(r'_\d{8}$', '', stem) # 去掉末尾 _YYYYMMDD
    return name.strip()


# ──────────────────────────────────────────────
# 2. 读取法律文本
# ──────────────────────────────────────────────
def load_law_text(filepath: str) -> str:
    """读取法律文本，兼容 UTF-8 / GBK 编码"""
    for enc in ('utf-8', 'gbk', 'utf-8-sig'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码文件: {filepath}")


# ──────────────────────────────────────────────
# 3. 用 Qwen 模型回答问题
# ──────────────────────────────────────────────
def load_qwen_model(model_path: str):
    """加载本地 Qwen 模型（需要 GPU，约占 ~16GB 显存）"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"[INFO] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def ask_model(model, tokenizer, law_name: str) -> str:
    """向模型提问，获取回答"""
    question = f"{law_name}的主要内容是什么？请详细说明其核心条款和适用范围。"
    response, _ = model.chat(tokenizer, question, history=[])
    return response.strip()


# ──────────────────────────────────────────────
# 4. 构建单条 LoRA 数据
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一名专业的中国法律助手，请严格依据真实法律条文回答问题，"
    "不得编造或混淆法律内容。"
)

def build_sample(law_name: str, model_answer: str, real_text: str) -> dict:
    """
    构造 alpaca 格式的训练样本。
    instruction = 用户问题
    input       = 模型的错误/不完整回答（作为参考对比，可选）
    output      = 真实法律文本（训练目标）
    """
    instruction = (
        f"请介绍《{law_name}》的主要内容，包括立法目的、核心条款和适用范围。"
    )

    # 截取真实法律文本（避免超长；可调整）
    max_real_len = 3000
    real_excerpt = real_text[:max_real_len]
    if len(real_text) > max_real_len:
        real_excerpt += "\n\n（以下内容已省略，请参考完整法律文本）"

    return {
        "system":      SYSTEM_PROMPT,
        "instruction": instruction,
        "input":       "",          # 留空；如需放模型回答作对比可填 model_answer
        "output":      real_excerpt,
        # 调试用字段，训练时可忽略
        "_meta": {
            "law_name":     law_name,
            "model_answer": model_answer[:500],   # 仅记录前500字
        }
    }


# ──────────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--law_dir',    default='./lawtxtversion', help='法律文本文件夹')
    parser.add_argument('--output',     default='./lora_dataset.json', help='输出数据集路径')
    parser.add_argument('--model_path', default='Qwen/Qwen-7B-Chat',  help='Qwen 模型路径或 HF id')
    parser.add_argument('--skip_model', action='store_true',
                        help='跳过模型推理，直接用真实法条生成数据集（无显卡时使用）')
    parser.add_argument('--max_files',  type=int, default=None, help='调试用：只处理前N个文件')
    args = parser.parse_args()

    law_dir = Path(args.law_dir)
    if not law_dir.exists():
        raise FileNotFoundError(f"找不到法律文件夹: {law_dir}")

    # 收集所有 .txt 文件
    txt_files = sorted(law_dir.glob('*.txt'))
    if args.max_files:
        txt_files = txt_files[:args.max_files]
    print(f"[INFO] 找到 {len(txt_files)} 个法律文本文件")

    # 加载模型
    model, tokenizer = None, None
    if not args.skip_model:
        model, tokenizer = load_qwen_model(args.model_path)

    dataset = []
    skipped = []

    for fpath in tqdm(txt_files, desc="处理法律文件"):
        law_name = parse_law_name(fpath.name)

        try:
            real_text = load_law_text(str(fpath))
        except Exception as e:
            print(f"\n[WARN] 读取失败 {fpath.name}: {e}")
            skipped.append(fpath.name)
            continue

        # 获取模型回答
        if model and tokenizer:
            try:
                model_answer = ask_model(model, tokenizer, law_name)
            except Exception as e:
                print(f"\n[WARN] 模型推理失败 {law_name}: {e}")
                model_answer = ""
        else:
            model_answer = ""  # skip_model 模式

        sample = build_sample(law_name, model_answer, real_text)
        dataset.append(sample)

    # 保存数据集
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据集已保存: {output_path}")
    print(f"   总样本数: {len(dataset)}")
    if skipped:
        print(f"   跳过文件: {skipped}")


if __name__ == '__main__':
    main()