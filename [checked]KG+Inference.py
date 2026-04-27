"""
Legal KG Inference Script (多法律文件输出)
根据案情检索最相似的5个法律知识图谱，并分别执行多跳推理
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# =============================
# Config (与训练时保持一致)
# =============================
CHROMA_PATH = "./chroma_db"
COLLECTION = "china_law"
MODEL_NAME = "shibing624/text2vec-base-chinese"
MODEL_DIR = "./models"
KG_DIR = "./kg_store"
EMB_DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================
# EmbeddingManager (增加返回ids)
# =============================
class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(name=COLLECTION)

    def encode(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def search(self, text, k=5):
        emb = self.encode(text)
        res = self.collection.query(
            query_embeddings=[emb.tolist()],
            n_results=k
        )
        return res["documents"][0], res["metadatas"][0], res["ids"][0]


# =============================
# KG 类
# =============================
class KG:
    def __init__(self, name):
        self.name = name
        self.triplets = []
        self.entity2id = {}
        self.id2entity = {}
        self.rel2id = {}
        self.id2rel = {}

    def load_from_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.name = data["name"]
        self.triplets = data["triplets"]
        self.entity2id = data["entity2id"]
        self.rel2id = data["rel2id"]
        self.id2entity = {int(v): k for k, v in self.entity2id.items()}
        self.id2rel = {int(v): k for k, v in self.rel2id.items()}

    def indexed(self):
        return [(self.entity2id[h], self.rel2id[r], self.entity2id[t])
                for h, r, t in self.triplets]


# =============================
# CompGCN 模型定义
# =============================
class CompGCNLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_self = torch.nn.Linear(dim, dim)
        self.W_nei = torch.nn.Linear(dim, dim)
        self.W_rel = torch.nn.Linear(dim, dim)
        self.act = torch.nn.ReLU()

    def forward(self, ent, rel, edge_index, edge_type):
        src, dst = edge_index
        r_emb = rel[edge_type]
        msg = ent[src] * torch.sigmoid(r_emb)
        agg = torch.zeros_like(ent)
        agg.index_add_(0, dst, msg)
        out = self.W_self(ent) + self.W_nei(agg)
        out = self.act(out)
        rel = self.W_rel(rel)
        return out, rel


class CompGCN(torch.nn.Module):
    def __init__(self, num_ent, num_rel, init_emb):
        super().__init__()
        self.ent = torch.nn.Embedding.from_pretrained(init_emb, freeze=False)
        self.rel = torch.nn.Embedding(num_rel, EMB_DIM)
        self.layers = torch.nn.ModuleList([
            CompGCNLayer(EMB_DIM),
            CompGCNLayer(EMB_DIM)
        ])

    def encode(self, edge_index, edge_type):
        x = self.ent.weight
        r = self.rel.weight
        for layer in self.layers:
            x, r = layer(x, r, edge_index, edge_type)
        return x, r

    def score(self, h, r, t):
        return (h * r * t).sum(-1)


# =============================
# 图构建函数
# =============================
def build_graph(kg):
    src, dst, rel = [], [], []
    for h, r, t in kg.indexed():
        src += [h, t]
        dst += [t, h]
        rel += [r, r]
    return (
        torch.tensor([src, dst], device=DEVICE),
        torch.tensor(rel, device=DEVICE)
    )


# =============================
# 多跳推理函数
# =============================
def get_subgraph(kg, seeds, hops=2):
    nodes = set(seeds)
    edges = []
    for _ in range(hops):
        for h, r, t in kg.triplets:
            if h in nodes or t in nodes:
                edges.append((h, r, t))
                nodes.add(h)
                nodes.add(t)
    return edges


def find_paths(edges, start, depth=3):
    paths = []

    def dfs(cur, nodes, rels, d):
        if d >= depth:
            return
        for h, r, t in edges:
            if h == cur:
                paths.append((nodes + [t], rels + [r]))
                dfs(t, nodes + [t], rels + [r], d + 1)

    for s in start:
        dfs(s, [s], [], 0)
    return paths


def score_paths(paths, model, kg, edge_index, edge_type):
    with torch.no_grad():
        ent, rel = model.encode(edge_index, edge_type)

    scored = []
    for nodes, rels in paths:
        score = 0
        valid = True
        for i in range(len(rels)):
            h = kg.entity2id.get(nodes[i])
            t = kg.entity2id.get(nodes[i + 1])
            r = kg.rel2id.get(rels[i])
            if h is None or t is None or r is None:
                valid = False
                break
            score += model.score(
                ent[h].unsqueeze(0),
                rel[r].unsqueeze(0),
                ent[t].unsqueeze(0)
            ).item()
        if valid:
            scored.append((score, nodes, rels))
    return sorted(scored, reverse=True)[:5]


def build_reasoning(case, scored):
    out = f"案情：{case}\n\n推理链：\n"
    for s, nodes, rels in scored:
        chain = ""
        for i in range(len(rels)):
            chain += f"{nodes[i]} --[{rels[i]}]--> "
        chain += nodes[-1]
        out += f"{chain} (score={s:.2f})\n"
    return out


# =============================
# 加载模型函数
# =============================
def load_model_and_kg(law_name):
    """根据法律文件名称加载对应的模型和KG"""
    model_path = os.path.join(MODEL_DIR, f"{law_name}_compgcn.pt")
    kg_path = os.path.join(KG_DIR, f"{law_name}_kg.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"KG文件不存在: {kg_path}")

    # 加载KG
    kg = KG(law_name)
    kg.load_from_file(kg_path)

    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # 初始化模型（使用零嵌入占位，实际权重从checkpoint恢复）
    init_emb = torch.zeros((len(kg.entity2id), EMB_DIM), dtype=torch.float32)
    model = CompGCN(len(kg.entity2id), len(kg.rel2id), init_emb).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"✅ 成功加载模型和KG: {law_name}")
    return model, kg


# =============================
# 从检索结果中提取唯一的法律文件名称
# =============================
def extract_unique_doc_names(metas, ids, k):
    """从检索结果中提取去重后的法律文件名称列表（保持顺序）"""
    seen = set()
    doc_names = []
    for i in range(k):
        if metas and "doc" in metas[i]:
            doc = metas[i]["doc"]
        else:
            doc = ids[i].rsplit('_', 1)[0]
        if doc not in seen:
            seen.add(doc)
            doc_names.append(doc)
    return doc_names


# =============================
# 主推理流程（支持多个法律文件）
# =============================
def infer_from_case(case_text, top_k=5):
    """输入案情文本，对最相似的top_k个法律文件分别输出推理结果"""
    emb_mgr = EmbeddingManager()
    logger.info(f"正在检索最相似的 {top_k} 个实体...")
    docs, metas, ids = emb_mgr.search(case_text, k=top_k)

    # 提取去重后的法律文件名称
    doc_names = extract_unique_doc_names(metas, ids, top_k)
    logger.info(f"检索到 {len(doc_names)} 个唯一法律文件: {doc_names}")

    # 对每个法律文件执行推理
    for idx, law_name in enumerate(doc_names, 1):
        print(f"\n{'='*60}")
        print(f"📚 法律文件 {idx}/{len(doc_names)}: {law_name}")
        print('='*60)

        try:
            model, kg = load_model_and_kg(law_name)
        except FileNotFoundError as e:
            logger.error(f"加载失败: {e}")
            continue

        # 构建图
        edge_index, edge_type = build_graph(kg)

        # 检索种子实体（与该法律文件相关的前5个实体）
        seeds = emb_mgr.search(case_text, k=5)[0]

        # 子图与路径搜索
        subgraph = get_subgraph(kg, seeds, hops=2)
        paths = find_paths(subgraph, seeds, depth=3)
        scored_paths = score_paths(paths, model, kg, edge_index, edge_type)

        # 输出推理结果
        reasoning_output = build_reasoning(case_text, scored_paths)
        print(reasoning_output)


# =============================
# 示例运行
# =============================
if __name__ == "__main__":
    # 测试案情
    test_case = "行为人多次骗取他人财物"
    infer_from_case(test_case, top_k=5)