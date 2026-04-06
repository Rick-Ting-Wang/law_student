"""
法律知识图谱 + CompGCN + Chroma Embedding
修复版本：适配新 Chroma API + 正确的 embedding 维度
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import chromadb

# ════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════
CHROMA_PATH = "./chroma_db"
COLLECTION = "china_law"
MODEL_NAME = "shibing624/text2vec-base-chinese"

TRIPLET_DIR = "./3entity"
KG_OUTPUT_DIR = "./legal_kg"
INFERENCE_OUT_DIR = "./kg_inference"
VIZ_DIR = "./kg_visualization"

# GNN 超参
EMBEDDING_DIM = 768  # ✅ 改成 768（text2vec-base-chinese 的真实维度）
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NEG_SAMPLE_N = 5
MARGIN = 1.0
DROPOUT = 0.3
NUM_GCN_LAYERS = 2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# Chroma + SentenceTransformer 初始化（新 API）
# ════════════════════════════════════════════════════════════
class ChromaEmbedding:
    """管理 Chroma 和 text2vec embedding - 新 API 版本"""

    def __init__(self, model_name: str, chroma_path: str, collection_name: str):
        logger.info(f"Loading SentenceTransformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ Embedding dimension: {self.embedding_dim}")

        # ✅ 新 Chroma API（简化初始化）
        logger.info(f"Initializing Chroma at: {chroma_path}")
        os.makedirs(chroma_path, exist_ok=True)

        try:
            # 新 API：直接用 HttpClient 或内存模式
            self.client = chromadb.PersistentClient(path=chroma_path)
            logger.info("✅ Using new PersistentClient API")
        except Exception as e:
            logger.warning(f"PersistentClient failed, using EphemeralClient: {e}")
            self.client = chromadb.EphemeralClient()

        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✅ Collection ready: {collection_name}")

    def embed_text(self, text: str) -> np.ndarray:
        """文本转 embedding"""
        return self.model.encode(text, convert_to_numpy=True)

    def add_entities_to_chroma(self, entities: list, doc_name: str):
        """添加实体到 Chroma"""
        ids = [f"{doc_name}#{e}" for e in entities]
        embeddings = [self.embed_text(e).tolist() for e in entities]
        metadatas = [{"doc": doc_name, "entity": e} for e in entities]

        self.collection.add(
            ids=ids,
            documents=entities,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"✅ Added {len(entities)} entities to Chroma: {doc_name}")

    def get_entity_embeddings(self, entities: list) -> torch.Tensor:
        """获取实体的 embedding"""
        embeddings = []
        for entity in entities:
            emb = self.embed_text(entity)
            embeddings.append(emb)
        return torch.tensor(np.array(embeddings), dtype=torch.float32)


# ════════════════════════════════════════════════════════════
# 知识图谱数据结构
# ════════════════════════════════════════════════════════════
class LocalKG:
    def __init__(self, doc_name: str):
        self.doc_name = doc_name
        self.triplets = []
        self.entity2id = {}
        self.id2entity = {}
        self.rel2id = {}
        self.id2rel = {}

    def add_triplets(self, raw: list):
        for item in raw:
            h = item['subject'].strip()
            r = item['relation'].strip()
            t = item['object'].strip()
            if not h or not r or not t:
                continue
            self.triplets.append((h, r, t))

    def build_vocab(self):
        entities = sorted({h for h, r, t in self.triplets} |
                          {t for h, r, t in self.triplets})
        relations = sorted({r for h, r, t in self.triplets})
        self.entity2id = {e: i for i, e in enumerate(entities)}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.rel2id = {r: i for i, r in enumerate(relations)}
        self.id2rel = {i: r for r, i in self.rel2id.items()}

    @property
    def indexed_triplets(self):
        return [(self.entity2id[h], self.rel2id[r], self.entity2id[t])
                for h, r, t in self.triplets]

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.rel2id)


def load_all_kgs(triplet_dir: str) -> dict:
    kgs = {}
    for jf in Path(triplet_dir).glob("*.json"):
        doc = jf.stem
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f"Skip {jf}: {e}")
            continue
        raw = data.get('triplets', [])
        if not raw:
            continue
        kg = LocalKG(doc)
        kg.add_triplets(raw)
        kg.build_vocab()
        if kg.num_entities >= 2 and kg.num_relations >= 1:
            kgs[doc] = kg
            logger.info(f"  ✅ {doc}: entities={kg.num_entities} "
                       f"relations={kg.num_relations} triplets={len(kg.triplets)}")
    logger.info(f"Total valid KGs: {len(kgs)}")
    return kgs


# ════════════════════════════════════════════════════════════
# 负采样 Dataset
# ════════════════════════════════════════════════════════════
class KGDataset(Dataset):
    def __init__(self, triplets: list, num_entities: int, neg_n: int = 5):
        self.triplets = triplets
        self.num_entities = num_entities
        self.neg_n = neg_n
        self.pos_set = set(map(tuple, triplets))

    def __len__(self):
        return len(self.triplets) * self.neg_n

    def __getitem__(self, idx):
        pos = self.triplets[idx // self.neg_n]
        h, r, t = pos

        for _ in range(50):
            if random.random() < 0.5:
                h_neg = random.randint(0, self.num_entities - 1)
                neg = (h_neg, r, t)
            else:
                t_neg = random.randint(0, self.num_entities - 1)
                neg = (h, r, t_neg)
            if neg not in self.pos_set:
                break

        return (torch.tensor(h, dtype=torch.long),
                torch.tensor(r, dtype=torch.long),
                torch.tensor(t, dtype=torch.long),
                torch.tensor(neg[0], dtype=torch.long),
                torch.tensor(neg[1], dtype=torch.long),
                torch.tensor(neg[2], dtype=torch.long))


# ════════════════════════════════════════════════════════════
# CompGCN 层
# ════════════════════════════════════════════════════════════
class CompGCNLayer(nn.Module):
    """CompGCN 层 - 用 Chroma embedding 初始化"""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 dropout: float = 0.2):
        super().__init__()
        self.W_rel = nn.Linear(in_dim, out_dim, bias=False)
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neighbor = nn.Linear(in_dim, out_dim, bias=False)

        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, ent_emb, rel_emb, edge_index, edge_type):
        """
        ent_emb  : [N_e, D] - 来自 Chroma (768 维)
        rel_emb  : [N_r, D]
        edge_index: [2, E]
        edge_type : [E]
        """
        N = ent_emb.size(0)
        src, dst = edge_index

        # 获取关系 embedding
        r_emb = rel_emb[edge_type]  # [E, D]
        src_emb = ent_emb[src]      # [E, D]

        # 关系调制
        neighbor_msg = src_emb * torch.sigmoid(r_emb)  # [E, D]

        # 聚合：平均池化
        agg = torch.zeros(N, ent_emb.size(1), device=ent_emb.device)
        count = torch.zeros(N, 1, device=ent_emb.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(neighbor_msg), neighbor_msg)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(len(dst), 1, device=ent_emb.device))
        count = count.clamp(min=1)
        agg = agg / count

        # 融合：自身 + 邻居
        out = self.W_self(ent_emb) + self.W_neighbor(agg)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)

        # 关系 embedding 变换
        new_rel = self.W_rel(rel_emb)
        return out, new_rel


# ════════════════════════════════════════════════════════════
# CompGCN 模型
# ════════════════════════════════════════════════════════════
class CompGCN(nn.Module):
    """
    CompGCN - 用 Chroma embedding 初始化
    ✅ entity_emb 来自 text2vec-base-chinese (768维)
    ✅ relation_emb 仍随机初始化
    """

    def __init__(self, num_entities: int, num_relations: int,
                 entity_embeddings: torch.Tensor,
                 emb_dim: int = 768, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        # ✅ 用 Chroma embedding 初始化实体
        self.ent_emb = nn.Embedding(num_entities, emb_dim)
        self.ent_emb.weight = nn.Parameter(entity_embeddings)
        self.ent_emb.weight.requires_grad = True  # 允许微调

        # 关系 embedding 随机初始化
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.layers = nn.ModuleList([
            CompGCNLayer(emb_dim, emb_dim, num_relations, dropout)
            for _ in range(num_layers)
        ])

        self.drop = nn.Dropout(dropout)

    def encode(self, edge_index, edge_type):
        """编码所有实体和关系"""
        x = self.ent_emb.weight
        rel = self.rel_emb.weight
        for layer in self.layers:
            x, rel = layer(x, rel, edge_index, edge_type)
        return x, rel

    def score(self, h_emb, r_emb, t_emb):
        """DistMult 打分"""
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def forward(self, h_idx, r_idx, t_idx, edge_index, edge_type):
        """前向传播"""
        ent_emb, rel_emb = self.encode(edge_index, edge_type)
        h = self.drop(ent_emb[h_idx])
        r = self.drop(rel_emb[r_idx])
        t = self.drop(ent_emb[t_idx])
        return self.score(h, r, t)


# ════════════════════════════════════════════════════════════
# 训练函数
# ════════════════════════════════════════════════════════════
def build_graph_tensors(kg: LocalKG, device):
    """构建图的边和关系类型"""
    src, dst, rtype = [], [], []
    for h, r, t in kg.indexed_triplets:
        src.append(h)
        dst.append(t)
        rtype.append(r)
        # 反向边
        src.append(t)
        dst.append(h)
        rtype.append(r)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type = torch.tensor(rtype, dtype=torch.long, device=device)
    return edge_index, edge_type


def train_compgcn(kg: LocalKG, chroma_embedding_mgr: ChromaEmbedding,
                  device: torch.device):
    """
    训练 CompGCN（用 Chroma embedding 初始化）
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {kg.doc_name}")
    logger.info(f"{'='*60}")

    # ✅ 获取实体的 Chroma embedding
    entity_list = [kg.id2entity[i] for i in range(kg.num_entities)]
    entity_embs = chroma_embedding_mgr.get_entity_embeddings(entity_list)
    entity_embs = entity_embs.to(device)

    logger.info(f"  Entity embeddings shape: {entity_embs.shape}")

    # ✅ 用 Chroma embedding 初始化模型
    model = CompGCN(kg.num_entities, kg.num_relations,
                    entity_embeddings=entity_embs,
                    emb_dim=EMBEDDING_DIM,
                    num_layers=NUM_GCN_LAYERS,
                    dropout=DROPOUT).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MarginRankingLoss(margin=MARGIN)

    edge_index, edge_type = build_graph_tensors(kg, device)

    dataset = KGDataset(kg.indexed_triplets, kg.num_entities, neg_n=NEG_SAMPLE_N)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model.train()
    logger.info(f"  📊 entities={kg.num_entities} | relations={kg.num_relations} | "
                f"triplets={len(kg.triplets)} | epochs={NUM_EPOCHS}")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0

        for batch in dataloader:
            h_pos, r_pos, t_pos, h_neg, r_neg, t_neg = [b.to(device) for b in batch]

            score_pos = model(h_pos, r_pos, t_pos, edge_index, edge_type)
            score_neg = model(h_neg, r_neg, t_neg, edge_index, edge_type)

            y = torch.ones_like(score_pos)
            loss = criterion(score_pos, score_neg, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / max(len(dataloader), 1)
            logger.info(f"    Epoch {epoch:>3}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

    model.eval()
    logger.info(f"  ✅ Training complete: {kg.doc_name}")

    # ✅ 将实体添加到 Chroma
    chroma_embedding_mgr.add_entities_to_chroma(entity_list, kg.doc_name)

    return model


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Device: {device}")

    for d in [KG_OUTPUT_DIR, INFERENCE_OUT_DIR, VIZ_DIR]:
        os.makedirs(d, exist_ok=True)

    # ✅ 初始化 Chroma embedding（新 API）
    logger.info("\n" + "="*60)
    logger.info("Step 1: Initialize Chroma + Embedding Model")
    logger.info("="*60)
    chroma_mgr = ChromaEmbedding(MODEL_NAME, CHROMA_PATH, COLLECTION)

    # 加载知识图谱
    logger.info("\n" + "="*60)
    logger.info("Step 2: Load Knowledge Graphs")
    logger.info("="*60)
    kgs = load_all_kgs(TRIPLET_DIR)

    if not kgs:
        logger.error("No valid KGs found!")
        return

    # 训练
    logger.info("\n" + "="*60)
    logger.info("Step 3: Train CompGCN with Chroma Embeddings")
    logger.info("="*60)

    all_models = {}
    for doc_name, kg in kgs.items():
        try:
            model = train_compgcn(kg, chroma_mgr, device)
            all_models[doc_name] = model

            # 保存模型
            model_path = os.path.join(KG_OUTPUT_DIR, f"{doc_name}_compgcn_chroma.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"  💾 Model saved: {model_path}")
        except Exception as e:
            logger.error(f"  ❌ Error training {doc_name}: {e}")
            continue

    logger.info("\n" + "="*60)
    logger.info("✅ ALL DONE - Training Complete")
    logger.info("="*60)
    logger.info(f"Trained {len(all_models)} models")
    logger.info(f"Chroma DB: {CHROMA_PATH}")
    logger.info(f"Models: {KG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
