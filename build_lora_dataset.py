"""
build_lora_dataset.py（简化版）
输入：lawtxtversion 目录下的法律文本文件
输出：LoRA 训练用的 instruction dataset
- 问题：文件名中的法律名称是什么？
- 答案：法律文件的完整内容
"""

import json
import re
import os
from pathlib import Path

SYSTEM_PROMPT = "你是一名专业的中国法律助手，请严格依据真实法律条文回答问题，不得编造或混淆法律内容。"


def extract_law_name(filename):
    """
    从文件名中提取法律名称，去掉时间戳
    例如：中华人民共和国网络安全法20181001 -> 中华人民共和国网络安全法
    """
    # 去掉文件扩展名
    name = filename.replace('.txt', '')

    # 去掉末尾的数字时间戳（6-8位数字）
    name = re.sub(r'\d{6,8}$', '', name)

    # 去掉可能残留的末尾数字
    name = re.sub(r'\d+$', '', name)

    return name


def read_law_file(filepath):
    """
    读取法律文件内容
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content
    except Exception as e:
        print(f"读取失败 {filepath}: {e}")
        return None


def build_dataset(input_dir: str, output_path: str):
    """
    构建数据集
    """
    input_path = Path(input_dir)

    # 获取所有 txt 文件
    txt_files = list(input_path.glob('*.txt'))

    if not txt_files:
        print(f"未在 {input_dir} 中找到 txt 文件")
        return

    print(f"找到 {len(txt_files)} 个法律文件")

    dataset = []

    for txt_file in txt_files:
        # 提取法律名称
        law_name = extract_law_name(txt_file.stem)

        # 读取法律内容
        law_content = read_law_file(txt_file)

        if not law_content:
            print(f"跳过 {txt_file.name}（内容为空或读取失败）")
            continue

        # 构建训练样本
        sample = {
            "system": SYSTEM_PROMPT,
            "instruction": f"《{law_name}》是什么？",
            "input": "",
            "output": law_content
        }

        dataset.append(sample)
        print(f"✓ 已处理: {law_name}")

    # 保存数据集
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 成功生成 {len(dataset)} 条训练样本")
    print(f"✅ 保存至: {output_path}")

    # 打印示例
    if dataset:
        print("\n示例数据:")
        print("-" * 50)
        example = dataset[0]
        print(f"System: {example['system'][:50]}...")
        print(f"Instruction: {example['instruction']}")
        print(f"Output 长度: {len(example['output'])} 字符")
        print(f"Output 预览: {example['output'][:100]}...")


def main():
    # 配置路径
    INPUT_DIR = r"./lawtxtversion"
    OUTPUT_PATH = r"./lora_dataset.json"

    build_dataset(INPUT_DIR, OUTPUT_PATH)


if __name__ == "__main__":
    main()