"""
build_lora_dataset.py（结构化版）
输入：lora_structured_json（按条文拆分的 JSON）
输出：LoRA 训练用的 instruction dataset
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

SYSTEM_PROMPT = "你是一名专业的中国法律助手，请严格依据真实法律条文回答问题，不得编造或混淆法律内容。"


def load_structured(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset(records: list) -> list:
    dataset = []

    # ── 策略 1：单条文问答（精确查询） ──────────────────────────────────────
    for r in records:
        # Q: 某法第X条规定了什么？
        dataset.append({
            "system":      SYSTEM_PROMPT,
            "instruction": f"《{r['law']}》{r['article']}规定了什么？",
            "input":       "",
            "output":      r["text"],
        })
        # Q: chunk 原文（直接给 chunk 作为上下文让模型复述/解释）
        dataset.append({
            "system":      SYSTEM_PROMPT,
            "instruction": f"请解释以下法律条文的含义：\n{r['chunk']}",
            "input":       "",
            "output":      f"根据《{r['law']}》{r['article']}，{r['text']}",
        })

    # ── 策略 2：章节汇总（同一章的条文合并） ────────────────────────────────
    by_chapter = defaultdict(list)
    for r in records:
        key = (r["law"], r["chapter"])
        by_chapter[key].append(r)

    for (law, chapter), arts in by_chapter.items():
        combined = "\n\n".join(
            f"{a['article']}\n{a['text']}" for a in arts
        )
        dataset.append({
            "system":      SYSTEM_PROMPT,
            "instruction": f"请介绍《{law}》{chapter}的全部内容。",
            "input":       "",
            "output":      combined,
        })

    # ── 策略 3：整部法律汇总（按 law 分组） ─────────────────────────────────
    by_law = defaultdict(list)
    for r in records:
        by_law[r["law"]].append(r)

    for law, arts in by_law.items():
        combined = "\n\n".join(
            f"{a['chapter']} {a['article']}\n{a['text']}" for a in arts
        )
        dataset.append({
            "system":      SYSTEM_PROMPT,
            "instruction": f"请介绍《{law}》的全部内容。",
            "input":       "",
            "output":      combined,
        })

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./law_structured.json")
    parser.add_argument("--output", default="./lora_dataset.json")
    args = parser.parse_args()

    records = load_structured(args.input)
    print(f"读入 {len(records)} 条条文")

    dataset = build_dataset(records)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ 生成 {len(dataset)} 条训练样本 → {args.output}")


if __name__ == "__main__":
    main()