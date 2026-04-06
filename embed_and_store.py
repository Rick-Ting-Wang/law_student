import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path


# ================================================================
# 配置
# ================================================================

JSON_PATH   = "law_structured.json"
CHROMA_PATH = "./chroma_db"
COLLECTION  = "china_law"
MODEL_NAME  = "shibing624/text2vec-base-chinese"
BATCH_SIZE  = 64


# ================================================================
# 向量化 + 存库
# ================================================================

def build_vector_store():

    # ── 1. 读取 JSON ─────────────────────────────────────────────
    print("📂 读取 JSON...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   共 {len(data)} 个 chunk")

    # ── 2. 加载 Embedding 模型 ───────────────────────────────────
    print(f"\n🤖 加载模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ── 3. 初始化 ChromaDB ───────────────────────────────────────
    print(f"\n🗄️  初始化向量库：{CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 已存在就删掉重建（重新跑的时候用）
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"   已删除旧集合 [{COLLECTION}]，重新构建")

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # ── 4. 分批向量化 + 存入 ────────────────────────────────────
    print(f"\n⚙️  开始向量化，batch_size={BATCH_SIZE}...")

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="存入向量库"):

        batch = data[i: i + BATCH_SIZE]

        ids = [f"chunk_{i + j}" for j in range(len(batch))]

        # chunk 字段作为向量化文本（含法律名/章节/条文的完整上下文）
        documents = [item["chunk"] for item in batch]

        metadatas = [
            {
                "law":          item.get("law", ""),
                "publish_date": item.get("publish_date", ""),
                "chapter":      item.get("chapter", ""),
                "article":      item.get("article", ""),
                # text 单独存，方便检索后直接取原文
                # chroma metadata 有长度限制，截断保险
                "text":         item.get("text", "")[:1000],
            }
            for item in batch
        ]

        embeddings = model.encode(
            documents,
            show_progress_bar=False,
            batch_size=BATCH_SIZE
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    total = collection.count()
    print(f"\n✅ 完成！向量库共 {total} 条记录")


if __name__ == "__main__":
    build_vector_store()
