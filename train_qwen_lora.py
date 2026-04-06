import torch
import json
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================
# 配置
# =========================
MODEL_NAME = "Qwen/Qwen-7B-Chat"
DATA_PATH = "./lora_dataset.json"
OUTPUT_DIR = "./output_qwen_lora"

# =========================
# 1. 加载 tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)
tokenizer.pad_token_id = tokenizer.eod_id

# =========================
# 2. 加载模型
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# =========================
# 3. LoRA配置
# =========================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "c_proj"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)

# =========================
# 4. 数据处理函数（适配你的json）
# =========================
def process_func(example):
    MAX_LENGTH = 1024

    system = example.get("system", "")
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    prompt = f"<|im_start|>system\n{system}<|im_end|>\n" \
             f"<|im_start|>user\n{instruction}{input_text}<|im_end|>\n" \
             f"<|im_start|>assistant\n"

    response = f"{output_text}<|im_end|>\n"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)
    response_ids = tokenizer(response, add_special_tokens=False)

    input_ids = prompt_ids["input_ids"] + response_ids["input_ids"]
    attention_mask = prompt_ids["attention_mask"] + response_ids["attention_mask"]

    labels = [-100] * len(prompt_ids["input_ids"]) + response_ids["input_ids"]

    # 截断
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# =========================
# 5. 加载数据
# =========================
dataset = load_dataset("json", data_files=DATA_PATH)
dataset = dataset["train"].map(process_func, remove_columns=dataset["train"].column_names)

# =========================
# 6. loss记录 callback
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
# 7. 训练参数
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    save_total_limit=2,
    report_to="none"
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
trainer.train()

# =========================
# 10. 保存 LoRA 权重
# =========================
model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

# =========================
# 11. 保存 loss
# =========================
with open(f"{OUTPUT_DIR}/loss_log.json", "w") as f:
    json.dump(loss_log, f, indent=2)

print("✅ loss 已保存")

# =========================
# 12. 绘制 loss 曲线
# =========================
steps = [x["step"] for x in loss_log]
losses = [x["loss"] for x in loss_log]

plt.figure()
plt.plot(steps, losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
print("✅ loss 曲线已保存")