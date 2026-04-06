import torch
import json
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================
# 配置
# =========================
MODEL_NAME = "Qwen/Qwen-7B-Chat"
DATA_PATH = "./lora_dataset.json"
OUTPUT_DIR = "./output_qwen_qlora"

# =========================
# 1. tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)
tokenizer.pad_token_id = tokenizer.eod_id

# =========================
# 2. 4bit量化配置（核心）
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,   # 可以改成 bf16（更稳）
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# =========================
# 3. 加载模型（QLoRA方式）
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 开启梯度检查点（省显存）
model.gradient_checkpointing_enable()

# =========================
# 4. LoRA配置
# =========================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "c_proj"],
    r=4,                 # 👈 降低rank（更省显存）
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)

# =========================
# 5. 数据处理（防爆token）
# =========================
def process_func(example):
    MAX_LENGTH = 1024   # 👈 控制显存关键

    system = example.get("system", "")
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # 🚨 关键：截断超长法律文本
    output_text = output_text[:2000]

    prompt = f"<|im_start|>system\n{system}<|im_end|>\n" \
             f"<|im_start|>user\n{instruction}{input_text}<|im_end|>\n" \
             f"<|im_start|>assistant\n"

    response = f"{output_text}<|im_end|>\n"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)
    response_ids = tokenizer(response, add_special_tokens=False)

    input_ids = prompt_ids["input_ids"] + response_ids["input_ids"]
    attention_mask = prompt_ids["attention_mask"] + response_ids["attention_mask"]

    labels = [-100] * len(prompt_ids["input_ids"]) + response_ids["input_ids"]

    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# =========================
# 6. 数据集
# =========================
dataset = load_dataset("json", data_files=DATA_PATH)
dataset = dataset["train"].map(process_func, remove_columns=dataset["train"].column_names)

# =========================
# 7. loss记录
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
# 8. 训练参数（极限省显存）
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # 👈 拉大，减少显存
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

# =========================
# 9. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[LossLogger()]
)

# =========================
# 10. 开始训练
# =========================
trainer.train()

# =========================
# 11. 保存模型
# =========================
model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

# =========================
# 12. 保存loss
# =========================
with open(f"{OUTPUT_DIR}/loss_log.json", "w") as f:
    json.dump(loss_log, f, indent=2)

# =========================
# 13. 可视化
# =========================
steps = [x["step"] for x in loss_log]
losses = [x["loss"] for x in loss_log]

plt.figure()
plt.plot(steps, losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("QLoRA Training Loss")
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
print("✅ 训练完成 + loss曲线已保存")