import torch
import json
import matplotlib.pyplot as plt
import os

# 🔧 关键：设置环境变量减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================
# 配置
# =========================
MODEL_NAME = "Qwen/Qwen-7B-Chat"
DATA_PATH = "./lora_dataset.json"
OUTPUT_DIR = "./output_qwen_lora"

# =========================
# 1. tokenizer（优化）
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)
tokenizer.pad_token_id = tokenizer.eod_id

# =========================
# 2. 加载模型（使用 8bit 量化减少显存）
# =========================
print("正在加载模型...")

try:
    from transformers import BitsAndBytesConfig

    # 8bit 量化配置（大幅减少显存）
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",  # 自动分配设备
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # 减少 CPU 内存使用
    )
    print("✅ 启动")

except ImportError:
    print("⚠️ 未安装 bitsandbytes，使用 FP16 模式（需要更多显存）")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

# 开启梯度检查点（省显存）
model.gradient_checkpointing_enable()

# =========================
# 3. LoRA 配置（减少参数量）
# =========================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "c_proj"],
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数量


# =========================
# 4. 数据处理（截断更短）
# =========================
def process_func(example):
    MAX_LENGTH = 512

    system = example.get("system", "")
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # 进一步截断输出
    output_text = output_text[:1000]

    prompt = f"<|im_start|>system\n{system}<|im_end|>\n" \
             f"<|im_start|>user\n{instruction}{input_text}<|im_end|>\n" \
             f"<|im_start|>assistant\n"

    response = f"{output_text}<|im_end|>\n"

    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    response_ids = tokenizer(response, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)

    input_ids = prompt_ids["input_ids"] + response_ids["input_ids"]
    attention_mask = prompt_ids["attention_mask"] + response_ids["attention_mask"]
    labels = [-100] * len(prompt_ids["input_ids"]) + response_ids["input_ids"]

    # 截断到最大长度
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# =========================
# 5. 数据集（使用更少数据）
# =========================
dataset = load_dataset("json", data_files=DATA_PATH)
dataset = dataset["train"].map(process_func, remove_columns=dataset["train"].column_names)

# 🔧 先用小数据集测试（100条）
USE_SMALL_DATASET = False  # 设为 False 使用全部数据
if USE_SMALL_DATASET:
    dataset = dataset.select(range(min(100, len(dataset))))
    print(f"⚠️ 使用小数据集: {len(dataset)} 条样本（测试用）")
else:
    print(f"使用全部数据集: {len(dataset)} 条样本")

# =========================
# 6. loss 记录
# =========================
loss_log = []


class LossLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss_log.append({
                "step": state.global_step,
                "loss": logs["loss"]
            })


# =========================
# 7. 训练参数（优化显存）
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # 保持 1
    gradient_accumulation_steps=8,  # 👈 从 16 减到 8
    num_train_epochs=1,  # 👈 从 2 减到 1
    learning_rate=2e-4,
    logging_steps=5,
    save_steps=100,
    fp16=True,
    bf16=False,
    save_total_limit=1,  # 只保留 1 个 checkpoint
    report_to="none",
    dataloader_num_workers=0,  # 👈 改为 0，避免多进程问题
    gradient_checkpointing=True,
    optim="adamw_8bit",  # 👈 使用 8bit 优化器节省显存
    max_grad_norm=0.3,
    warmup_ratio=0.03,
)

# =========================
# 8. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[LossLogger()]
)

# =========================
# 9. 开始训练
# =========================
print("开始训练...")
trainer.train()

# =========================
# 10. 保存模型
# =========================
model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

# =========================
# 11. 保存 loss
# =========================
with open(f"{OUTPUT_DIR}/loss_log.json", "w") as f:
    json.dump(loss_log, f, indent=2)

# =========================
# 12. 可视化
# =========================
if loss_log:
    steps = [x["step"] for x in loss_log]
    losses = [x["loss"] for x in loss_log]

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("LoRA Training Loss")
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
    print("✅ 训练完成 + loss 曲线已保存")
else:
    print("⚠️ 没有记录到 loss 数据")