import os
import json
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model

# ==============================
# 配置信息
# ==============================

# ✅ 使用您本地的模型路径
MODEL_PATH = r"D:\hf_models\hub\models--Qwen--Qwen-7B-Chat\snapshots\93a65d34827a3cc269b727e67004743b723e2f83"
DATA_PATH = r"D:\Law_Student-master\Law_Student-master\data_train.json"
OUTPUT_DIR = "./output/Qwen-Law-LoRA"

# 系统提示词
SYSTEM_PROMPT = (
    "你是一名专业的中国刑事法律助手，擅长根据案件事实进行罪名认定、"
    "法条适用与量刑预测。请根据用户提供的案件事实，给出相关法条、"
    "罪名以及刑期的判断结果。"
)


# ==============================
# 1️⃣  数据加载与预处理
# ==============================

def load_jsonl(path: str) -> list:
    """读取 JSON Lines 格式的数据文件"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_output(meta: dict) -> str:
    """将元数据转换为中文输出"""
    articles = "、".join(str(a) for a in meta.get("relevant_articles", []))
    accusations = "、".join(meta.get("accusation", []))
    criminals = "、".join(meta.get("criminals", []))

    toi = meta.get("term_of_imprisonment", {})
    if toi.get("death_penalty"):
        sentence = "死刑"
    elif toi.get("life_imprisonment"):
        sentence = "无期徒刑"
    else:
        months = toi.get("imprisonment", 0)
        sentence = f"有期徒刑{months}个月"

    fine = meta.get("punish_of_money", 0)
    fine_str = f"，并处罚金{fine}元" if fine > 0 else ""

    return (
        f"【相关法条】第{articles}条\n"
        f"【罪名】{accusations}\n"
        f"【被告人】{criminals}\n"
        f"【量刑结果】{sentence}{fine_str}"
    )


def convert_to_instruction_format(records: list) -> pd.DataFrame:
    """转换为 instruction-input-output 格式"""
    rows = []
    for r in records:
        rows.append({
            "instruction": "请根据以下案件事实，判断适用法条、罪名及量刑结果：",
            "input": r["fact"],
            "output": build_output(r["meta"]),
        })
    return pd.DataFrame(rows)


# ==============================
# 2️⃣  Tokenization
# ==============================

def process_func(example, tokenizer):
    """处理单个样本"""
    MAX_LENGTH = 1024  # 法律文本可能较长

    # 构造输入提示
    prompt_parts = [
        "<|im_start|>system",
        SYSTEM_PROMPT + "<|im_end|>",
        "<|im_start|>user",
        example["instruction"] + example["input"] + "<|im_end|>",
        "",
    ]
    prompt_text = "\n".join(prompt_parts).strip()

    # Tokenize 输入部分
    instruction_enc = tokenizer(
        prompt_text + "\n",
        add_special_tokens=False,
    )

    # Tokenize 输出部分（模型需要生成的部分）
    response_enc = tokenizer(
        "<|im_start|>assistant\n" + example["output"] + "<|im_end|>\n",
        add_special_tokens=False,
    )

    # 拼接 input_ids 和 attention_mask
    input_ids = (
            instruction_enc["input_ids"]
            + response_enc["input_ids"]
            + [tokenizer.pad_token_id]
    )
    attention_mask = (
            instruction_enc["attention_mask"]
            + response_enc["attention_mask"]
            + [1]
    )

    # 构造 labels：输入部分为 -100（不参与损失计算），输出部分参与损失
    labels = (
            [-100] * len(instruction_enc["input_ids"])
            + response_enc["input_ids"]
            + [tokenizer.pad_token_id]
    )

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ==============================
# 3️⃣  LoRA 配置
# ==============================

def setup_lora_config():
    """设置 LoRA 参数"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # Qwen 的注意力投影层名称
        target_modules=["c_attn", "c_proj", "w1", "w2"],
        inference_mode=False,
        r=8,  # LoRA 秩
        lora_alpha=32,  # LoRA 缩放因子
        lora_dropout=0.1,  # Dropout 比例
    )
    return lora_config


# ==============================
# 4️⃣  训练配置 - 修改为完整训练
# ==============================

def setup_training_args(output_dir: str):
    """设置训练参数 - 完整训练"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # 单卡批次大小
        gradient_accumulation_steps=4,  # 梯度累积步数（有效批次 = 4*4 = 16）
        num_train_epochs=3,  # ✅ 训练3个完整epoch（根据需求调整）
        learning_rate=1e-4,  # 学习率
        weight_decay=0.01,  # 权重衰减
        warmup_ratio=0.1,  # ✅ 使用warmup比例（10%的steps）
        logging_steps=10,  # ✅ 每10步打印日志
        save_strategy="epoch",  # ✅ 每个epoch保存一次
        save_total_limit=3,  # ✅ 保留最后3个检查点
        evaluation_strategy="no",  # 如果没有验证集，设为no
        gradient_checkpointing=True,  # 梯度检查点（节省显存）
        fp16=True,  # 混合精度训练
        dataloader_pin_memory=True,  # 加速数据加载
        dataloader_num_workers=4,  # ✅ 增加数据加载进程数
        report_to="none",  # 不使用wandb
        remove_unused_columns=False,  # 保留所有列
        optim="paged_adamw_32bit",  # 使用8-bit Adam优化器
        lr_scheduler_type="cosine",  # ✅ 使用余弦学习率调度
        load_best_model_at_end=False,  # 如果没有验证集，设为False
        metric_for_best_model="loss",  # 监控loss
        greater_is_better=False,  # loss越小越好
    )
    return training_args


# ==============================
# 5️⃣  主训练流程
# ==============================

def main():
    print("\n" + "=" * 80)
    print("🚀 开始 Qwen-7B-Chat LoRA 微调（完整训练）")
    print("=" * 80 + "\n")

    # ✅ 验证路径
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：模型路径不存在: {MODEL_PATH}")
        exit(1)

    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：数据路径不存在: {DATA_PATH}")
        exit(1)

    print(f"✅ 模型路径: {MODEL_PATH}")
    print(f"✅ 数据路径: {DATA_PATH}\n")

    # ─────────────────────────────────────────
    # 第 1 步：加载数据
    # ─────────────────────────────────────────
    print("📚 第 1 步：加载数据...")
    raw_records = load_jsonl(DATA_PATH)
    df = convert_to_instruction_format(raw_records)
    ds = Dataset.from_pandas(df)

    # ✅ 打印数据集信息
    print(f"   ✅ 加载 {len(ds)} 条数据")
    print(f"   📊 数据集预览：")
    print(f"      - instruction: {ds[0]['instruction'][:50]}...")
    print(f"      - input length: {len(ds[0]['input'])} 字符")
    print(f"      - output length: {len(ds[0]['output'])} 字符\n")

    # ─────────────────────────────────────────
    # 第 2 步：加载 Tokenizer
    # ─────────────────────────────────────────
    print("🔤 第 2 步：加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.padding_side = "right"
    print(f"   ✅ Tokenizer 加载成功")
    print(f"   📊 词汇表大小: {len(tokenizer)}\n")

    # ─────────────────────────────────────────
    # 第 3 步：Tokenize 数据集
    # ─────────────────────────────────────────
    print("⚡ 第 3 步：Tokenize 数据集...")
    tokenized_ds = ds.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=ds.column_names,
        num_proc=4,  # 使用 4 个进程加速
        desc="Tokenizing",
    )

    # ✅ 统计tokenized后的信息
    sample = tokenized_ds[0]
    print(f"   ✅ Tokenization 完成")
    print(f"   📊 样本统计：")
    print(f"      - input_ids 长度: {len(sample['input_ids'])}")
    print(f"      - labels 长度: {len(sample['labels'])}")
    print(f"      - 有效token数: {sum(1 for l in sample['labels'] if l != -100)}\n")

    # ─────────────────────────────────────────
    # 第 4 步：加载模型
    # ─────────────────────────────────────────
    print("🤖 第 4 步：加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,  # 混合精度
    )
    model.enable_input_require_grads()  # 梯度检查点需要

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"   ✅ 模型加载成功")
    print(f"   📊 模型参数统计：")
    print(f"      - 总参数: {total_params:.2f}B")
    print(f"      - 可训练参数: {trainable_params:.2f}M\n")

    # ─────────────────────────────────────────
    # 第 5 步：应用 LoRA
    # ─────────────────────────────────────────
    print("🔧 第 5 步：应用 LoRA...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()

    # ─────────────────────────────────────────
    # 第 6 步：设置训练参数
    # ─────────────────────────────────────────
    print("⚙️  第 6 步：准备训练...")
    training_args = setup_training_args(OUTPUT_DIR)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ✅ 计算总训练步数
    total_steps = len(tokenized_ds) * training_args.num_train_epochs // (
                training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * 0.1)

    print(f"   ✅ 输出目录: {OUTPUT_DIR}")
    print(f"   📊 训练配置统计：")
    print(f"      - 训练轮数: {training_args.num_train_epochs}")
    print(f"      - 总训练步数: ~{total_steps} 步")
    print(f"      - 预热步数: {warmup_steps} 步")
    print(f"      - 批次大小: {training_args.per_device_train_batch_size}")
    print(f"      - 梯度累积: {training_args.gradient_accumulation_steps}")
    print(
        f"      - 有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}\n")

    # ─────────────────────────────────────────
    # 第 7 步：创建 Trainer 并开始训练
    # ─────────────────────────────────────────
    print("🎯 第 7 步：开始训练...\n")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        ),
        # 可选：添加早停回调
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 训练！
    train_result = trainer.train()

    print("=" * 80)
    print(f"\n✅ 训练完成！")
    print(f"   - 最终损失: {train_result.training_loss:.4f}")
    print(f"   - 训练步数: {train_result.global_step}")
    print(f"   - 检查点保存在: {OUTPUT_DIR}\n")

    # ─────────────────────────────────────────
    # 第 8 步：保存最终模型
    # ─────────────────────────────────────────
    print("💾 第 8 步：保存最终模型...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

    # 保存训练状态
    with open(f"{OUTPUT_DIR}/training_state.json", "w", encoding="utf-8") as f:
        json.dump({
            "final_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "epoch": train_result.epoch,
            "total_steps": total_steps,
        }, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 模型已保存到: {OUTPUT_DIR}/final_model")
    print(f"   ✅ 训练状态已保存\n")

    # ─────────────────────────────────────────
    # 第 9 步：推理测试
    # ─────────────────────────────────────────
    print("🧪 第 9 步：推理测试...\n")
    test_fact = (
        "被告人李某因与邻居王某发生口角，拿起菜刀将王某的右臂砍伤。"
        "经鉴定，王某右臂损伤为轻伤二级。李某事后主动投案自首。"
    )

    test_prompt = "请根据以下案件事实，判断适用法条、罪名及量刑结果：" + test_fact

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取 assistant 部分
    if "<|im_start|>assistant" in result:
        result = result.split("<|im_start|>assistant")[-1]
        result = result.replace("<|im_end|>", "").strip()

    print(f"📝 测试问题：{test_fact}\n")
    print(f"💬 模型回答：\n{result}\n")

    print("=" * 80)
    print("🎉 全部完成！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()