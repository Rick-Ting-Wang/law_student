import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ==============================
# 向量库配置
# ==============================

CHROMA_PATH = "./chroma_db"
COLLECTION  = "china_law"
EMBED_MODEL = "shibing624/text2vec-base-chinese"


# ==============================
# Qwen 模型配置
# ==============================

LLM_MODEL = "Qwen/Qwen-7B-Chat"


# ==============================
# Law Retriever
# ==============================

class LawRetriever:

    def __init__(self):

        self.model = SentenceTransformer(EMBED_MODEL)

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(COLLECTION)

        print(f"✅ 向量库加载完成，共 {self.collection.count()} 条法律记录")

    def search(self, query, top_k=5, law_filter=None):

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
                "score": round(1 - dist, 4),
                "law": meta["law"],
                "article": meta["article"],
                "chapter": meta["chapter"],
                "text": meta["text"],
            })

        return hits


# ==============================
# 构造 RAG Prompt
# ==============================

def build_prompt(question, law_hits):

    context = ""

    for i, hit in enumerate(law_hits):

        context += f"""
【参考法律{i+1}】

法律名称：{hit["law"]}
章节：{hit["chapter"]}
条款：{hit["article"]}

内容：
{hit["text"]}

"""

    prompt = f"""

请根据给定的法律条文回答问题。

====================
法律条文
====================

{context}

====================
问题
====================

{question}

====================
回答
====================
"""

    return prompt


# ==============================
# 加载 Qwen
# ==============================

def load_llm():

    print("🚀 Loading Qwen-7B-Chat...")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    print("✅ LLM loaded")

    return tokenizer, model


# ==============================
# LLM 推理
# ==============================

def ask_llm(prompt, tokenizer, model):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.1
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result


# ==============================
# 测试问题
# ==============================

TEST_QUESTIONS = [

    "金融机构违反实名制规定会受到什么处罚？",

    "生产转基因生物需要办理哪些许可？",

    "学校是否必须开展法治教育？",

    "企业污染环境会承担什么法律责任？",

    "非法集资会被如何处罚？"

]


# ==============================
# 主程序
# ==============================

def main():

    retriever = LawRetriever()

    tokenizer, model = load_llm()

    for question in TEST_QUESTIONS:

        print("\n" + "="*80)
        print("❓ 问题:", question)
        print("="*80)

        # 1 检索
        hits = retriever.search(question, top_k=5)
        # 2 构造 prompt
        prompt = build_prompt(question, hits)

        # 3 LLM 推理
        answer = ask_llm(prompt, tokenizer, model)

        print("\n🤖 模型回答：\n")
        print(answer)


if __name__ == "__main__":
    main()