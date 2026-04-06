import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "./chroma_db"
COLLECTION  = "china_law"
MODEL_NAME  = "shibing624/text2vec-base-chinese"


class LawRetriever:

    def __init__(self):
        # ⚠️ 必须和建库时用同一个模型
        self.model = SentenceTransformer(MODEL_NAME)

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(COLLECTION)

        print(f"✅ 向量库已加载，共 {self.collection.count()} 条记录")

    def search(
        self,
        query:      str,
        top_k:      int = 5,
        law_filter: str = None,   # 可选：只在某部法律内搜索
    ) -> list[dict]:

        query_vector = self.model.encode(query).tolist()

        where = {"law": law_filter} if law_filter else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "score":        round(1 - dist, 4),   # 余弦相似度，越高越好
                "law":          meta["law"],
                "publish_date": meta["publish_date"],
                "chapter":      meta["chapter"],
                "article":      meta["article"],
                "text":         meta["text"],
                "chunk":        doc,
            })

        return hits


# ── 本地测试 ────────────────────────────────────────────────────
if __name__ == "__main__":

    retriever = LawRetriever()

    queries = [
        "金融机构违反实名制规定会受到什么处罚？",
        "转基因生物生产需要哪些许可证？",
        "学校应当如何开展法治宣传教育？",
    ]

    for q in queries:
        print(f"\n{'='*55}")
        print(f"❓ {q}")
        print(f"{'='*55}")

        hits = retriever.search(q, top_k=8)

        for i, h in enumerate(hits, 1):
            print(f"\n  [{i}] 相似度 {h['score']:.2%}")
            print(f"       {h['law']} · {h['chapter']} · {h['article']}")
            print(f"       {h['text'][:80]}...")
