"""
train_lora.py
使用 LoRA 微调 Qwen-7B-Chat（适配 RTX 5060 Ti 16GB / CUDA 12.8）
不依赖 bitsandbytes

用法：
    python train_lora.py
    python train_lora.py --data ./lora_dataset.json --output ./qwen-lora-law

依赖：
    pip install transformers peft accelerate datasets quanto
"""

import json
import argparse
import logging

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_LENGTH     = 512
LORA_R         = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["c_attn", "c_proj", "w1", "w2"]


def load_dataset_from_json(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"读入 {len(records)} 条训练样本")
    return Dataset.from_list(records)


def build_prompt(system: str, instruction: str, input_text: str) -> str:
    user_content = instruction if not input_text else f"{instruction}\n{input_text}"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def tokenize_batch(batch, tokenizer, max_length: int) -> dict:
    input_ids_list  = []
    labels_list     = []
    attention_masks = []

    for system, instruction, inp, output in zip(
        batch["system"],
        batch["instruction"],
        batch["input"],
        batch["output"],
    ):
        prompt    = build_prompt(system, instruction, inp)
        full_text = prompt + output + "<|im_end|>"

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_enc   = tokenizer(
            full_text,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        input_ids  = full_enc["input_ids"]
        attn_mask  = full_enc["attention_mask"]
        prompt_len = len(prompt_ids)

        if prompt_len >= len(input_ids):
            logger.warning("样本 prompt 超出 max_length，已跳过")
            continue

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks.append(attn_mask)

    return {
        "input_ids":      input_ids_list,
        "labels":         labels_list,
        "attention_mask": attention_masks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="Qwen/Qwen-7B-Chat")
    parser.add_argument("--data",    default="./lora_dataset.json")
    parser.add_argument("--output",  default="./qwen-lora-law")
    parser.add_argument("--epochs",  type=int, default=3)
    parser.add_argument("--max_len", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )
    # Qwen 的 pad_token 需要这样强制设置
    tokenizer.pad_token    = "<|endoftext|>"
    tokenizer.pad_token_id = tokenizer.eod_id   # Qwen 专用 eod_id

    # ── 2. 加载模型 ───────────────────────────────────────────────────────────
    logger.info("加载模型（fp16，直接到 GPU）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        fp16=True,
    )
    model.config.use_cache = False

    # ── 3. quanto int8 量化 ───────────────────────────────────────────────────
    logger.info("应用 quanto int8 量化...")
    try:
        from quanto import quantize, freeze, qint8
        quantize(model, weights=qint8)
        freeze(model)
        logger.info("✅ quanto int8 量化完成")
    except ImportError:
        logger.warning("quanto 未安装，以 fp16 全精度运行（显存约 14 GB）")
        logger.warning("如需量化：pip install quanto")

    # ── 4. 注入 LoRA ──────────────────────────────────────────────────────────
    logger.info(f"注入 LoRA（r={LORA_R}, alpha={LORA_ALPHA}）...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. 数据预处理 ─────────────────────────────────────────────────────────
    logger.info("预处理数据集...")
    raw_ds = load_dataset_from_json(args.data)
    tokenized_ds = raw_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_len),
        batched=True,
        batch_size=64,
        remove_columns=raw_ds.column_names,
        desc="Tokenizing",
    )
    logger.info(f"有效样本数：{len(tokenized_ds)}")

    # ── 6. 训练参数 ───────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="adamw_torch",

        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        fp16=True,
        bf16=False,

        logging_steps=10,
        save_steps=200,
        save_total_limit=2,

        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    # ── 关键修复：把 pad_token_id 显式传给 collator ───────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=collator,
    )

    # ── 7. 训练 ───────────────────────────────────────────────────────────────
    logger.info("开始训练...")
    trainer.train()

    # ── 8. 保存 ───────────────────────────────────────────────────────────────
    logger.info(f"保存 LoRA 权重至 {args.output} ...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("✅ 训练完成！")


if __name__ == "__main__":
    main()