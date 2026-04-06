"""
build_chroma_db_law.py
======================
将 lawtxtversion/ 目录下所有 .txt 法律文本向量化，
使用 Lawformer (thunlp/Lawformer) 作为 Embedding 模型，
存入本地 ChromaDB 持久化向量数据库。

Lawformer 是清华大学专为中文法律长文本训练的模型，
基于 Longformer 架构，最大支持 4096 tokens。

依赖安装：
    pip install chromadb transformers torch

用法：
    python build_chroma_db_law.py
    python build_chroma_db_law.py --law_dir ./lawtxtversion --db_dir ./chroma_law_db
"""

import argparse
import torch
import chromadb
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# ── 配置 ────────────────────────────────────────────────────────────────────
DEFAULT_LAW_DIR = "./lawtxtversion"
DEFAULT_DB_DIR  = "./chroma_law_db"
COLLECTION_NAME = "law_collection"

# Lawformer 官方模型（清华大学，中文法律专用）
LAWFORMER_MODEL = "thunlp/Lawformer"
# 若已本地下载，改为本地路径，例如：
# LAWFORMER_MODEL = "D:/models/Lawformer"

MAX_SEQ_LEN   = 4096  # Lawformer 基于 Longformer，支持最长 4096 tokens
CHUNK_SIZE    = 1500  # 每块字符数（中文约 1500 字 ≈ 2000 tokens，留余量）
CHUNK_OVERLAP = 150   # 相邻块重叠字符数，保证上下文连贯
BATCH_SIZE    = 2     # Lawformer 较大，显存不足时调小到 1
# ────────────────────────────────────────────────────────────────────────────


def load_model(model_name: str):
    """加载 Lawformer tokenizer 和模型。"""
    print(f"[1/4] 加载 Lawformer 模型：{model_name}")
    print("      首次运行会从 HuggingFace 下载模型，请耐心等待……")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"      模型加载完成，使用设备：{device}")
    return tokenizer, model, device


def mean_pooling(model_output, attention_mask):
    """对最后一层隐藏状态做均值池化，得到定长句子向量。"""
    token_embeddings = model_output.last_hidden_state        # (B, L, H)
    mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode_texts(texts: list, tokenizer, model, device) -> list:
    """将一批文本编码为 L2 归一化向量列表。"""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    output = model(**encoded)
    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().tolist()


def chunk_text(text: str) -> list:
    """
    把长文本切成若干块。
    Lawformer 支持 4096 tokens，中文约 1 字 ≈ 1.3 token，
    1500 字约 2000 tokens，留有安全余量。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_law_files(law_dir: str) -> list:
    """读取 lawtxtversion/ 下所有 .txt 文件。"""
    law_dir = Path(law_dir)
    if not law_dir.exists():
        raise FileNotFoundError(f"找不到法律文本目录：{law_dir.resolve()}")

    files = sorted(law_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"目录 {law_dir} 下没有找到任何 .txt 文件")

    print(f"[2/4] 找到 {len(files)} 个法律文本文件")
    laws = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = f.read_text(encoding="gbk").strip()
        laws.append({
            "filename": f.name,
            "name":     f.stem,
            "text":     text,
        })
    return laws


def build_database(law_dir: str, db_dir: str):
    """主流程：读文件 → 切块 → Lawformer 编码 → 存 ChromaDB。"""

    # ── 加载模型
    tokenizer, model, device = load_model(LAWFORMER_MODEL)

    # ── 读取法律文件
    laws = load_law_files(law_dir)

    # ── 初始化 ChromaDB
    print(f"[3/4] 初始化 ChromaDB，存储路径：{Path(db_dir).resolve()}")
    client = chromadb.PersistentClient(path=db_dir)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"      检测到已有 collection '{COLLECTION_NAME}'，删除并重建……")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── 向量化并写入
    print("[4/4] 开始向量化……\n")

    all_ids, all_embeddings, all_documents, all_metadatas = [], [], [], []

    for idx, law in enumerate(laws, 1):
        chunks = chunk_text(law["text"])
        print(f"  [{idx:>3}/{len(laws)}] {law['name'][:35]:<35} "
              f"{len(law['text']):>6} 字  →  {len(chunks)} 块")

        for i, chunk in enumerate(chunks):
            all_ids.append(f"{law['name']}__chunk_{i}")
            all_documents.append(chunk)
            all_metadatas.append({
                "law_name":     law["name"],
                "filename":     law["filename"],
                "chunk_index":  i,
                "total_chunks": len(chunks),
            })

        # 逐批编码
        for b in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[b: b + BATCH_SIZE]
            vecs = encode_texts(batch, tokenizer, model, device)
            all_embeddings.extend(vecs)

    # 分批写入 ChromaDB
    WRITE_BATCH = 50
    print(f"\n  写入 ChromaDB，共 {len(all_ids)} 个向量块……")
    for i in range(0, len(all_ids), WRITE_BATCH):
        collection.add(
            ids=all_ids[i: i + WRITE_BATCH],
            embeddings=all_embeddings[i: i + WRITE_BATCH],
            documents=all_documents[i: i + WRITE_BATCH],
            metadatas=all_metadatas[i: i + WRITE_BATCH],
        )

    print(f"\n✅ 构建完成！")
    print(f"   法律文件数  ：{len(laws)}")
    print(f"   向量块总数  ：{len(all_ids)}")
    print(f"   数据库位置  ：{Path(db_dir).resolve()}")
    print(f"\n现在可以运行 law_retriever.py 进行检索。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建法律文本 ChromaDB 向量数据库（Lawformer）")
    parser.add_argument("--law_dir", default=DEFAULT_LAW_DIR, help="法律 txt 文件夹路径")
    parser.add_argument("--db_dir",  default=DEFAULT_DB_DIR,  help="ChromaDB 持久化路径")
    args = parser.parse_args()

    build_database(args.law_dir, args.db_dir)