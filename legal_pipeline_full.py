"""
law_triple_printer_full.py
==========================
第一阶段：Lawformer 检索最相似的法律文件（内嵌 law_retriever 功能）
第二阶段：加载对应 CompGCN 模型，打印知识图谱三元组及模型评分

无需额外导入 law_retriever，所有代码自包含。
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import chromadb

# =============================
# 配置参数
# =============================
# 法律文本向量库（Lawformer）
LAW_DB_DIR = "./chroma_law_db"
LAW_COLLECTION = "law_collection"
LAWFORMER_MODEL = "thunlp/Lawformer"
MAX_SEQ_LEN = 4096
TOP_K_LAWS = 5          # 最终返回前5部法律
TOP_K_CHUNKS = 30       # 先多取一些块用于聚合

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
    def __init__(self, db_dir=LAW_DB_DIR, model_name=LAWFORMER_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[初始化] 加载 Lawformer：{model_name}  (设备: {self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[初始化] 模型加载完成")

        db_path = Path(db_dir)
        if not db_path.exists():
            raise FileNotFoundError(
                f"找不到 ChromaDB 目录：{db_path.resolve()}\n"
                "请先运行 build_chroma_db_law.py 构建向量数据库。"
            )
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_collection(LAW_COLLECTION)
        print(f"[初始化] ChromaDB 加载成功，共 {self.collection.count()} 个向量块")

    @torch.no_grad()
    def _encode(self, text: str) -> list:
        """将单条文本编码为 L2 归一化向量。"""
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        output = self.model(**encoded)

        # 均值池化
        token_emb = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        embedding = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        # L2 归一化
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().tolist()[0]

    def _aggregate_by_law(self, chroma_results: dict) -> list:
        """按法律名称聚合，返回去重排序后的前 TOP_K_LAWS 部法律。"""
        ids       = chroma_results["ids"][0]
        documents = chroma_results["documents"][0]
        metadatas = chroma_results["metadatas"][0]
        distances = chroma_results["distances"][0]

        law_best = {}
        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = round(1 - dist, 4)
            law_name = meta["law_name"]

            if law_name not in law_best or similarity > law_best[law_name]["similarity"]:
                law_best[law_name] = {
                    "law_name":     law_name,
                    "filename":     meta["filename"],
                    "similarity":   similarity,
                    "chunk_index":  meta["chunk_index"],
                    "total_chunks": meta["total_chunks"],
                    "matched_text": doc,
                }

        ranked = sorted(law_best.values(), key=lambda x: x["similarity"], reverse=True)
        return ranked[:TOP_K_LAWS]

    def search(self, query: str) -> list:
        """检索与 query 最相关的前 5 部法律。"""
        if not query.strip():
            raise ValueError("查询内容不能为空")

        query_vec = self._encode(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(TOP_K_CHUNKS, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return self._aggregate_by_law(results)


# =============================
# 2. KG 与 CompGCN 模型组件
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


def load_model_and_kg(law_name):
    model_path = os.path.join(MODEL_DIR, f"{law_name}_compgcn.pt")
    kg_path = os.path.join(KG_DIR, f"{law_name}_kg.json")

    if not os.path.exists(model_path) or not os.path.exists(kg_path):
        raise FileNotFoundError(f"模型或KG文件不存在: {law_name}")

    kg = KG(law_name)
    kg.load_from_file(kg_path)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    init_emb = torch.zeros((len(kg.entity2id), EMB_DIM), dtype=torch.float32)
    model = CompGCN(len(kg.entity2id), len(kg.rel2id), init_emb).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, kg


def print_all_triples_with_scores(model, kg, top_n=None):
    """计算并打印知识图谱中的所有三元组及其模型评分"""
    edge_index, edge_type = build_graph(kg)
    with torch.no_grad():
        ent, rel = model.encode(edge_index, edge_type)

    triples = kg.triplets
    scores = []
    for h, r, t in triples:
        if h in kg.entity2id and t in kg.entity2id and r in kg.rel2id:
            h_id = kg.entity2id[h]
            t_id = kg.entity2id[t]
            r_id = kg.rel2id[r]
            score = model.score(
                ent[h_id].unsqueeze(0),
                rel[r_id].unsqueeze(0),
                ent[t_id].unsqueeze(0)
            ).item()
            scores.append((score, h, r, t))

    scores.sort(key=lambda x: x[0], reverse=True)

    print(f"\n知识图谱三元组及模型评分 (共 {len(scores)} 条):")
    if top_n and top_n < len(scores):
        scores = scores[:top_n]
        print(f"显示前 {top_n} 条 (得分从高到低):")
    else:
        print("全部显示 (得分从高到低):")

    for s, h, r, t in scores:
        print(f"  {h} --[{r}]--> {t}  (得分: {s:.4f})")


# =============================
# 3. 主流程
# =============================
def main(query_text, print_triples_limit=None):
    print("\n" + "="*70)
    print("🔍 第一阶段：法律文本检索（Lawformer）")
    print("="*70)

    retriever = LawRetriever()
    top_laws = retriever.search(query_text)

    print(f"\n查询：{query_text}")
    print(f"\n最相关的前 {len(top_laws)} 部法律：")
    for rank, law_info in enumerate(top_laws, 1):
        print(f"\n  【{rank}】{law_info['law_name']}")
        print(f"       相似度：{law_info['similarity']:.4f}")
        print(f"       命中片段：{law_info['matched_text'][:150]}...")

    print("\n" + "="*70)
    print("🧠 第二阶段：知识图谱三元组及模型评分")
    print("="*70)

    for rank, law_info in enumerate(top_laws, 1):
        law_name = law_info["law_name"]
        print(f"\n📚 法律文件 [{rank}/{len(top_laws)}]：{law_name}")

        try:
            model, kg = load_model_and_kg(law_name)
        except FileNotFoundError as e:
            logger.warning(f"跳过 {law_name}：{e}")
            continue

        print_all_triples_with_scores(model, kg, top_n=print_triples_limit)


if __name__ == "__main__":
    # 示例查询
    query = "2019年11月15日21时许，被告人金德林与徐加泉各持猎枪相约打猎，被告人金德林将徐某2误认为猎物开枪致其死亡。"
    # 参数 print_triples_limit 控制每个法律打印多少条三元组（None表示全部）
    main(query, print_triples_limit=50)