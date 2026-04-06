"""
law_retriever.py
================
对用户输入的问题，使用 Lawformer 编码后，
在 ChromaDB 中检索最匹配的前 5 部法律，
返回法律名称 + 匹配片段 + 相似度分数。

请先运行 build_chroma_db_law.py 构建向量数据库。

依赖安装：
    pip install chromadb transformers torch

用法（命令行单次查询）：
    python law_retriever.py --query "故意伤害罪的量刑标准是什么？"

用法（交互模式，不传 query）：
    python law_retriever.py

用法（作为模块调用）：
    from law_retriever import LawRetriever
    retriever = LawRetriever()
    results = retriever.search("故意伤害罪的量刑标准是什么？")
    for r in results:
        print(r["law_name"], r["similarity"])
"""

import argparse
import torch
import chromadb
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# ── 配置（与 build_chroma_db_law.py 保持一致）────────────────────────────────
DEFAULT_DB_DIR  = "./chroma_law_db"
COLLECTION_NAME = "law_collection"

LAWFORMER_MODEL = "thunlp/Lawformer"
# 若已本地下载，改为本地路径，例如：
# LAWFORMER_MODEL = "D:/models/Lawformer"

MAX_SEQ_LEN  = 4096
TOP_K_LAWS   = 5    # 最终返回前 5 部法律
TOP_K_CHUNKS = 30   # 先多取一些块，再按法律聚合去重
# ────────────────────────────────────────────────────────────────────────────


class LawRetriever:
    """
    法律文本检索器。
    加载 Lawformer + ChromaDB，对输入问题编码后
    检索最相关的前 TOP_K_LAWS 部法律。
    """

    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        model_name: str = LAWFORMER_MODEL,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = self._load_model(model_name)
        self.collection = self._load_db(db_dir)

    # ── 内部方法 ────────────────────────────────────────────────────────────

    def _load_model(self, model_name: str):
        print(f"[初始化] 加载 Lawformer：{model_name}  (设备: {self.device})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        model.eval()
        print("[初始化] 模型加载完成")
        return tokenizer, model

    def _load_db(self, db_dir: str):
        db_path = Path(db_dir)
        if not db_path.exists():
            raise FileNotFoundError(
                f"找不到 ChromaDB 目录：{db_path.resolve()}\n"
                "请先运行 build_chroma_db_law.py 构建向量数据库。"
            )
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(COLLECTION_NAME)
        print(f"[初始化] ChromaDB 加载成功，共 {collection.count()} 个向量块")
        return collection

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
        """
        ChromaDB 返回块级别结果，按法律名称聚合：
        每部法律保留相似度最高的块作为代表，
        最终返回去重排序后的前 TOP_K_LAWS 部。
        """
        ids       = chroma_results["ids"][0]
        documents = chroma_results["documents"][0]
        metadatas = chroma_results["metadatas"][0]
        distances = chroma_results["distances"][0]

        # ChromaDB cosine distance ∈ [0, 2]，转换为相似度
        law_best: dict = {}
        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = round(1 - dist, 4)   # 越大越相似
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

    # ── 公开接口 ────────────────────────────────────────────────────────────

    def search(self, query: str) -> list:
        """
        检索与 query 最相关的前 5 部法律。

        返回列表，每项包含：
            law_name     法律名称（txt 文件名去掉扩展名）
            filename     原始文件名
            similarity   相似度 ∈ [0, 1]，越高越相关
            matched_text 最匹配的文本片段
            chunk_index  该片段是第几块
            total_chunks 该法律共切了多少块
        """
        if not query.strip():
            raise ValueError("查询内容不能为空")

        query_vec = self._encode(query)

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(TOP_K_CHUNKS, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        return self._aggregate_by_law(results)


# ── 输出格式化 ───────────────────────────────────────────────────────────────

def print_results(query: str, results: list):
    """格式化打印检索结果。"""
    print("\n" + "=" * 62)
    print(f"查询：{query}")
    print("=" * 62)
    print(f"最相关的前 {len(results)} 部法律：\n")

    for rank, r in enumerate(results, 1):
        bar_len = int(r["similarity"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        print(f"  [{rank}] {r['law_name']}")
        print(f"       相似度：{r['similarity']:.4f}  {bar}")
        print(f"       文件名：{r['filename']}")
        print(f"       命中片段（第 {r['chunk_index']+1} / {r['total_chunks']} 块）：")

        preview = r["matched_text"][:150].replace("\n", " ")
        if len(r["matched_text"]) > 150:
            preview += "……"
        print(f"       {preview}")
        print()

    print("=" * 62)


# ── 交互模式 ─────────────────────────────────────────────────────────────────

def interactive_mode(retriever: LawRetriever):
    """交互式命令行检索模式。"""
    print("\n法律检索系统已就绪（输入 q 退出）\n")
    while True:
        try:
            query = input("请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not query:
            continue
        if query.lower() in ("q", "quit", "exit", "退出"):
            print("退出。")
            break
        results = retriever.search(query)
        print_results(query, results)


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法律文本检索器（Lawformer + ChromaDB）")
    parser.add_argument("--query",  default="",              help="检索问题；不传则进入交互模式")
    parser.add_argument("--db_dir", default=DEFAULT_DB_DIR,  help="ChromaDB 路径")
    parser.add_argument("--model",  default=LAWFORMER_MODEL, help="Embedding 模型路径或名称")
    args = parser.parse_args()

    retriever = LawRetriever(db_dir=args.db_dir, model_name=args.model)

    if args.query:
        results = retriever.search(args.query)
        print_results(args.query, results)
    else:
        interactive_mode(retriever)