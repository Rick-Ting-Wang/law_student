"""
Legal Case Matcher + KG Reasoner
第一阶段：用 Lawformer 检索最相似的法律文本（chroma_law_db）
第二阶段：加载对应法律文件的 CompGCN 模型和 KG，进行多跳推理
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModel

# =============================
# 配置
# =============================
# 法律文本向量库（Lawformer）
LAW_DB_PATH = "./chroma_law_db"
LAW_COLLECTION = "law_collection"
LAWFORMER_MODEL = "thunlp/Lawformer"
LAW_TEXT_DIR = "./lawtxtversion"          # 原始法律文本存放目录

# 实体向量库（text2vec）及模型/知识图谱目录
ENTITY_DB_PATH = "./chroma_db"
ENTITY_COLLECTION = "china_law"
TEXT2VEC_MODEL = "shibing624/text2vec-base-chinese"
MODEL_DIR = "./models"
KG_DIR = "./kg_store"

EMB_DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================
# 1. 法律文本检索器（Lawformer）
# =============================
class LawRetriever:
    def __init__(self, db_path, collection_name, model_name, law_text_dir):
        # 加载 Lawformer 模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

        # 连接 ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

        # 原始法律文本目录
        self.law_text_dir = Path(law_text_dir)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode_query(self, text):
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        ).to(DEVICE)
        output = self.model(**encoded)
        emb = self.mean_pooling(output, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]

    def retrieve_top_laws(self, query_text, top_k=5):
        """返回最相似的 top_k 个法律文件名称及代表性文本片段"""
        q_emb = self.encode_query(query_text)
        results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k * 3  # 多取一些块，用于聚合
        )

        # 按法律文件聚合相似度得分
        law_scores = {}
        law_best_chunk = {}
        for doc, meta, distance in zip(results["documents"][0],
                                       results["metadatas"][0],
                                       results["distances"][0]):
            law_name = meta["law_name"]
            similarity = 1 - distance  # cosine distance 转 similarity
            if law_name not in law_scores:
                law_scores[law_name] = []
                law_best_chunk[law_name] = (doc, similarity)
            law_scores[law_name].append(similarity)
            # 保留相似度最高的文本块
            if similarity > law_best_chunk[law_name][1]:
                law_best_chunk[law_name] = (doc, similarity)

        # 计算每个法律的平均相似度并排序
        avg_scores = {law: np.mean(scores) for law, scores in law_scores.items()}
        sorted_laws = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 获取原始法律文本（可选，用于打印摘要）
        law_summaries = {}
        for law, _ in sorted_laws:
            law_file = self.law_text_dir / f"{law}.txt"
            if law_file.exists():
                text = law_file.read_text(encoding="utf-8")[:500] + "..."
                law_summaries[law] = text
            else:
                law_summaries[law] = "（原始文本文件未找到）"

        return sorted_laws, law_best_chunk, law_summaries


# =============================
# 2. 实体检索与 KG 推理组件（沿用原逻辑）
# =============================
class EntityRetriever:
    def __init__(self, db_path, collection_name, model_name):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

    def encode(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def search_entities(self, text, k=5):
        emb = self.encode(text)
        res = self.collection.query(
            query_embeddings=[emb.tolist()],
            n_results=k
        )
        return res["documents"][0]  # 返回实体名称列表


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
    out = f"\n案情：{case}\n\n推理链：\n"
    for s, nodes, rels in scored:
        chain = ""
        for i in range(len(rels)):
            chain += f"{nodes[i]} --[{rels[i]}]--> "
        chain += nodes[-1]
        out += f"{chain} (score={s:.2f})\n"
    return out


def load_model_and_kg(law_name):
    model_path = os.path.join(MODEL_DIR, f"{law_name}_compgcn.pt")
    kg_path = os.path.join(KG_DIR, f"{law_name}_kg.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"KG文件不存在: {kg_path}")

    kg = KG(law_name)
    kg.load_from_file(kg_path)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    init_emb = torch.zeros((len(kg.entity2id), EMB_DIM), dtype=torch.float32)
    model = CompGCN(len(kg.entity2id), len(kg.rel2id), init_emb).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, kg


# =============================
# 主流程
# =============================
def main(case_text, top_k_laws=3):
    print("\n" + "="*70)
    print("🔍 第一阶段：法律文本匹配（Lawformer）")
    print("="*70)

    # 初始化法律检索器
    law_retriever = LawRetriever(
        db_path=LAW_DB_PATH,
        collection_name=LAW_COLLECTION,
        model_name=LAWFORMER_MODEL,
        law_text_dir=LAW_TEXT_DIR
    )

    # 检索最相似的法律文件
    top_laws, best_chunks, summaries = law_retriever.retrieve_top_laws(case_text, top_k=top_k_laws)

    print(f"\n案情描述：{case_text[:100]}...")
    print(f"\n检索到 {len(top_laws)} 个最相似法律文件：")
    for rank, (law, avg_sim) in enumerate(top_laws, 1):
        print(f"\n  【{rank}】{law}  (平均相似度: {avg_sim:.4f})")
        print(f"       代表性片段：{best_chunks[law][0][:200]}...")
        # 可选：打印原始法律文本开头
        # print(f"       法律文本开头：{summaries[law][:150]}...")

    print("\n" + "="*70)
    print("🧠 第二阶段：知识图谱推理（CompGCN）")
    print("="*70)

    # 初始化实体检索器
    entity_retriever = EntityRetriever(
        db_path=ENTITY_DB_PATH,
        collection_name=ENTITY_COLLECTION,
        model_name=TEXT2VEC_MODEL
    )

    # 对每个匹配到的法律文件进行推理
    for rank, (law_name, _) in enumerate(top_laws, 1):
        print(f"\n📚 正在处理法律文件 [{rank}/{len(top_laws)}]：{law_name}")

        try:
            model, kg = load_model_and_kg(law_name)
        except FileNotFoundError as e:
            logger.warning(f"跳过 {law_name}：{e}")
            continue

        edge_index, edge_type = build_graph(kg)

        # 检索种子实体（基于 text2vec 模型）
        seeds = entity_retriever.search_entities(case_text, k=5)
        logger.info(f"种子实体：{seeds}")

        subgraph = get_subgraph(kg, seeds, hops=2)
        paths = find_paths(subgraph, seeds, depth=3)
        scored_paths = score_paths(paths, model, kg, edge_index, edge_type)

        print(build_reasoning(case_text, scored_paths))


if __name__ == "__main__":
    # 示例案情
    case = "2019年11月15日21时许，被告人金德林与徐加泉各持猎枪相约打猎，被告人金德林将徐某2误认为猎物开枪致其死亡。"
    # case = "行为人多次骗取他人财物"
    main(case, top_k_laws=3)