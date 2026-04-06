import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 原始模型
BASE_MODEL = r"D:\hf_models\hub\models--Qwen--Qwen-7B-Chat\snapshots\93a65d34827a3cc269b727e67004743b723e2f83"

# LoRA权重
LORA_PATH = r"./output/Qwen-Law-LoRA/final_model"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载LoRA
model = PeftModel.from_pretrained(model, LORA_PATH)

model.eval()

prompt = """
请根据以下案件事实，
被告人张某入室盗窃现金5000元，被抓获。
做法律分析
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))