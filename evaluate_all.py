"""
LexEval 一键全指标评测脚本
- 自动扫描 model_output/zero_shot/ 下所有模型目录
- 选择题同时计算 Accuracy 和 F1
- 生成题同时计算 Rouge_L（+ 可选 Bertscore / Bartscore）
- 结果汇总为一张宽表 CSV，每行一个模型×任务，每列一个指标
直接 python evaluate_all.py 运行，无需参数。
"""

import os
import re
import json
import string
import jieba
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from rouge import Rouge

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════
MODEL_OUTPUT_DIR = 'model_output/zero_shot'   # 扫描此目录下所有模型子目录
OUTPUT_CSV       = 'evaluation_result/evaluation_result_all.csv'

# 生成题额外指标（需要对应模型）
USE_BERTSCORE  = False          # 开启则需要填 BERT_MODEL_PATH
USE_BARTSCORE  = False          # 开启则需要填 BART_MODEL_PATH
BERT_MODEL_PATH = None          # e.g. '/path/to/chinese-roberta-wwm-ext'
BART_MODEL_PATH = None          # e.g. '/path/to/bart-large-cnn-zh'
DEVICE          = 'cuda'         # 'cpu' 或 'cuda'
# ════════════════════════════════════════════════════════════


# ── process.py 工具函数 ──────────────────────────────────────
def find_valid_substrings(s):
    s = s.split('解析')[0].split('分析')[0]
    s = s.replace("、","").replace(".","").replace(",","").replace(";","") \
         .replace("，","").replace("和","").replace(", ","")
    pattern = r'[ABCDE]{1,5}'
    substrings = re.findall(pattern, s)
    valid = [sub for sub in substrings if len(sub) == len(set(sub))]
    valid = "".join(valid)
    valid = ''.join(OrderedDict.fromkeys(valid))
    return valid


def normalize_zh_answer(s):
    def white_space_fix(text): return "".join(text.split())
    def remove_punc(text):
        cn_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        bad = set(string.punctuation + cn_punc)
        return "".join(ch for ch in text if ch not in bad)
    def lower(text): return text.lower()
    return white_space_fix(remove_punc(lower(s)))


# ── 数据加载 ─────────────────────────────────────────────────
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── 各指标计算（与 evaluate_specific.py 完全一致） ────────────────────
def eval_accuracy(data):
    score = sum(1 for d in data if find_valid_substrings(d['output']) == d['answer'])
    return round(score / len(data), 4) if data else 0.0


def eval_f1(data):
    scores = []
    for d in data:
        pred = set(find_valid_substrings(d['output']))
        gt   = set(d['answer'])
        p = len(pred & gt) / len(pred) if pred else 0
        r = len(pred & gt) / len(gt)   if gt   else 0
        scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def eval_rougel(data):
    rouge = Rouge()
    scores = []
    for d in data:
        pred = " ".join(list(jieba.cut(normalize_zh_answer(d['output']), cut_all=False)))
        ans  = " ".join(list(jieba.cut(normalize_zh_answer(d['answer']),  cut_all=False)))
        try:
            scores.append(rouge.get_scores([pred], [ans], avg=True)["rouge-l"]["f"])
        except:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def eval_bertscore(data, model_path, device, batch_size=10):
    import bert_score
    preds = [d['output'] for d in data]
    refs  = [d['answer'] for d in data]
    _, _, score_f1 = bert_score.score(
        preds, refs, lang='zh', verbose=False,
        model_type=model_path, num_layers=8,
        device=device, batch_size=batch_size
    )
    return round((sum(score_f1) / len(score_f1)).item(), 4)


def eval_bartscore(data, model_path, device, batch_size=10):
    import torch, torch.nn as nn
    from transformers import BertTokenizer, BartForConditionalGeneration

    class BARTScorer:
        def __init__(self, checkpoint, device):
            self.device = device
            self.max_length = 1024
            self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
            self.model.eval()
            self.model.to(device)
            self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
            self.lsm = nn.LogSoftmax(dim=1)

        def score(self, srcs, tgts, batch_size=50):
            score_list = []
            for i in range(0, len(srcs), batch_size):
                src_list, tgt_list = srcs[i:i+batch_size], tgts[i:i+batch_size]
                with torch.no_grad():
                    enc_src = self.tokenizer(src_list, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt')
                    enc_tgt = self.tokenizer(tgt_list, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt')
                    src_tokens = enc_src['input_ids'].to(self.device)
                    src_mask   = enc_src['attention_mask'].to(self.device)
                    tgt_tokens = enc_tgt['input_ids'].to(self.device)
                    tgt_len    = enc_tgt['attention_mask'].sum(dim=1).to(self.device)
                    output = self.model(input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens)
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1).sum(dim=1) / tgt_len
                    score_list += [-x.item() for x in loss]
            return score_list

    preds = [d['output'] for d in data]
    refs  = [d['answer'] for d in data]
    scorer = BARTScorer(checkpoint=model_path, device=device)
    scores = scorer.score(preds, refs, batch_size=batch_size)
    return round(sum(scores) / len(scores), 4)


# ── 单文件评估，返回指标字典 ─────────────────────────────────
def evaluate_file(file_path, task_name):
    data = load_jsonl(file_path)
    if not data:
        return {}

    metrics = {}
    is_generation = (task_name.split('_')[0] == '5')

    if is_generation:
        metrics['Rouge_L'] = eval_rougel(data)
        if USE_BERTSCORE and BERT_MODEL_PATH:
            metrics['Bertscore'] = eval_bertscore(data, BERT_MODEL_PATH, DEVICE)
        if USE_BARTSCORE and BART_MODEL_PATH:
            metrics['Bartscore'] = eval_bartscore(data, BART_MODEL_PATH, DEVICE)
    else:
        metrics['Accuracy'] = eval_accuracy(data)
        metrics['F1']       = eval_f1(data)

    return metrics


# ── 主流程 ───────────────────────────────────────────────────
def main():
    if not os.path.isdir(MODEL_OUTPUT_DIR):
        print(f"[ERROR] Directory not found: {MODEL_OUTPUT_DIR}")
        return

    # 扫描所有模型子目录
    all_models = sorted([
        d for d in os.listdir(MODEL_OUTPUT_DIR)
        if os.path.isdir(os.path.join(MODEL_OUTPUT_DIR, d))
    ])

    if not all_models:
        print(f"[ERROR] No model directories found in {MODEL_OUTPUT_DIR}")
        return

    print(f"Found {len(all_models)} model(s): {all_models}\n")

    rows = []

    for model in all_models:
        model_dir = os.path.join(MODEL_OUTPUT_DIR, model)
        all_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.jsonl')])

        if not all_files:
            print(f"  [SKIP] No .jsonl files in {model_dir}")
            continue

        print(f"  Evaluating: {model}  ({len(all_files)} files)")
        for f_name in tqdm(all_files, desc=f"  {model}"):
            # 文件名格式：{ModelName}_{task_num}_{sub}.jsonl → 取最后两段
            parts = f_name.replace('.jsonl', '').split('_')
            task_name = parts[-2] + '_' + parts[-1]

            # 合法性校验（与 evaluate_specific.py 一致）
            try:
                t_num, t_sub = int(task_name.split('_')[0]), int(task_name.split('_')[1])
                assert 1 <= t_num <= 6 and 1 <= t_sub <= 6
            except:
                print(f"    [SKIP] Cannot parse task name from: {f_name}")
                continue

            file_path = os.path.join(model_dir, f_name)
            metrics   = evaluate_file(file_path, task_name)

            row = {'model': model, 'task': task_name}
            row.update(metrics)
            rows.append(row)

    if not rows:
        print("No results to save.")
        return

    # 宽表：每行 model×task，指标各占一列，缺失值填 NaN
    df = pd.DataFrame(rows)
    # 固定列顺序
    fixed_cols = ['model', 'task']
    metric_cols = [c for c in df.columns if c not in fixed_cols]
    df = df[fixed_cols + sorted(metric_cols)]
    df = df.sort_values(['model', 'task']).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\nSaved → {OUTPUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()