import os, json, logging, http.client, glob, threading, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

API_KEYS = [
    "sk-fJ86ZwidviNgouL72e099cCb4f33491287B30eA005Ae362a",
    "sk-mQJTxHM9mn8Cr4pb87019f15132e41E4AaAaE591Eb175cD1",
    "sk-qxJAygZaLpEVrrnv2769C8843c4c45DbB422BcC42570F71a",
    "sk-31YmY2kfpSToD3kC3fB5C77f79B546A1A4569f919eC7165e",
    "sk-q03wvP3dTszR6jW7155dA7DbE5074934B1216987B27e6265",
    "sk-DWrhUWsxaqlAPED94eD140A81d4944228aA617E64f4a7c08",
    "sk-xvhC5NAaFWrFP20z6e3492Cf53594179A2D6FcDbCa2cE382",
    "sk-dq6R1LukORXZKAAF1aB54684Ef48472eA32eB8B98611E0E7",
    "sk-DSrGK9Z8rVHMSeqJ7d4dD052D65447058811D83eAeF65d60",
    "sk-SEtG1TnT8Fd3rhIBCf969808A3Fa43FcB2Ea84A4189aCe79",
]

API_HOST = "dpapi.cn"
MODEL_NAME = "deepseek-v3"

# ── 法律检索路径（text2vec，替换原 Lawformer）────────────────
CHROMA_LAW_PATH = "./chroma_db"
CHROMA_LAW_COLLECTION = "china_law"
TEXT2VEC_MODEL = "shibing624/text2vec-base-chinese"
TOP_K_LAWS = 1
TOP_K_CHUNKS = 30

# ── 知识图谱及模型路径 ──────────────────────────────────────
MODEL_DIR = "./models"
KG_DIR = "./kg_store"
EMB_DIM = 768
TOP_K_TRIPLES = 3
MAX_CHUNKS = 8

# ── 精确法条库 ──────────────────────────────────────────────
LAW_STRUCTURED_DB = "./law_structured.json"

# ── 数据与输出 ──────────────────────────────────────────────
DATA_DIR = "./data"
OUT_DIR = "./model_output/zero_shot/GNN_Chain_1_Step"
LOG_FILE = "gnn_chain_running.log"

# ── 重试 / 并发配置 ─────────────────────────────────────────
MAX_RETRY = 3
NUM_WORKERS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════
# 日志与 API Key 管理
# ════════════════════════════════════════════════════════════
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_tl = threading.local()
_key_lock = threading.Lock()
_key_idx = 0


def get_api_key():
    global _key_idx
    if not hasattr(_tl, "key"):
        with _key_lock:
            _tl.key = API_KEYS[_key_idx % len(API_KEYS)]
            _key_idx += 1
    return _tl.key


# ════════════════════════════════════════════════════════════
# 题目类型识别
# ════════════════════════════════════════════════════════════
def detect_question_type(task_name: str) -> str:
    """
    根据任务名识别题目类型

    返回值：
      - "choice"      选择题（原始逻辑）
      - "summary"     摘要题（5_1）
      - "judgment"    裁判分析（5_2）
      - "translate"   翻译题（5_3）
      - "essay"       论述题（5_4）
    """
    if "5_1" in task_name:
        return "summary"
    elif "5_2" in task_name:
        return "judgment"
    elif "5_3" in task_name:
        return "translate"
    elif "5_4" in task_name:
        return "essay"
    else:
        return "choice"


# ════════════════════════════════════════════════════════════
# 法律名称规范化
# ════════════════════════════════════════════════════════════
def normalize_law_name(raw: str) -> str:
    name = raw.strip()
    for p in ["中华人民共和国", "中国"]:
        if name.startswith(p):
            name = name[len(p):]
            break
    name = re.sub(r"[\s\-]+", "_", name)

    if os.path.exists(os.path.join(MODEL_DIR, f"{name}_compgcn.pt")):
        return name

    try:
        for fname in os.listdir(MODEL_DIR):
            if not fname.endswith("_compgcn.pt"):
                continue
            stem = fname[:-len("_compgcn.pt")]
            if name in stem or stem in name:
                return stem
    except FileNotFoundError:
        pass

    try:
        for fname in os.listdir(KG_DIR):
            if not fname.endswith("_kg.json"):
                continue
            stem = fname[:-len("_kg.json")]
            if name in stem or stem in name:
                return stem
    except FileNotFoundError:
        pass

    return name


def list_available_kg_names() -> list:
    try:
        return [f[:-len("_compgcn.pt")]
                for f in os.listdir(MODEL_DIR) if f.endswith("_compgcn.pt")]
    except FileNotFoundError:
        return []


# ════════════════════════════════════════════════════════════
# LawRetriever —— text2vec-base-chinese + ChromaDB
# ════════════════════════════════════════════════════════════
class LawRetriever:

    def __init__(self):
        print(f"[LawRetriever] Loading {TEXT2VEC_MODEL} ...")
        self.model = SentenceTransformer(TEXT2VEC_MODEL)

        db_path = Path(CHROMA_LAW_PATH)
        if not db_path.exists():
            raise FileNotFoundError(f"ChromaDB not found: {db_path}")
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_collection(CHROMA_LAW_COLLECTION)

        print(f"[LawRetriever] Ready — {self.collection.count()} chunks loaded")

    def _encode(self, text: str) -> list:
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def _aggregate(self, res: dict) -> list:
        best: dict = {}
        for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
        ):
            sim = round(1 - dist, 4)
            law_name = meta.get("law", "unknown")

            if law_name not in best or sim > best[law_name]["similarity"]:
                best[law_name] = {
                    "law_name": law_name,
                    "similarity": sim,
                    "matched_text": doc,
                    "article": meta.get("article", ""),
                    "chapter": meta.get("chapter", ""),
                    "publish_date": meta.get("publish_date", ""),
                }

        return sorted(
            best.values(), key=lambda x: x["similarity"], reverse=True
        )[:TOP_K_LAWS]

    def search(self, query: str) -> list:
        if not query.strip():
            return []
        vec = self._encode(query)
        res = self.collection.query(
            query_embeddings=[vec],
            n_results=min(TOP_K_CHUNKS, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return self._aggregate(res)


_law_retriever = None
_law_retriever_lock = threading.Lock()


def get_law_retriever() -> LawRetriever:
    global _law_retriever
    if _law_retriever is None:
        with _law_retriever_lock:
            if _law_retriever is None:
                _law_retriever = LawRetriever()
    return _law_retriever


# ════════════════════════════════════════════════════════════
# KG 数据结构与 CompGCN 模型
# ════════════════════════════════════════════════════════════
class KG:
    def __init__(self):
        self.name = ""
        self.triplets = []
        self.entity2id = {};
        self.id2entity = {}
        self.rel2id = {};
        self.id2rel = {}

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.name = d["name"]
        self.triplets = d["triplets"]
        self.entity2id = d["entity2id"]
        self.rel2id = d["rel2id"]
        self.id2entity = {int(v): k for k, v in self.entity2id.items()}
        self.id2rel = {int(v): k for k, v in self.rel2id.items()}

    def indexed(self):
        return [
            (self.entity2id[h], self.rel2id[r], self.entity2id[t])
            for h, r, t in self.triplets
            if h in self.entity2id and r in self.rel2id and t in self.entity2id
        ]


class CompGCNLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_self = torch.nn.Linear(dim, dim)
        self.W_nei = torch.nn.Linear(dim, dim)
        self.W_rel = torch.nn.Linear(dim, dim)
        self.act = torch.nn.ReLU()

    def forward(self, ent, rel, edge_index, edge_type):
        src, dst = edge_index
        msg = ent[src] * torch.sigmoid(rel[edge_type])
        agg = torch.zeros_like(ent)
        agg.index_add_(0, dst, msg)
        return self.act(self.W_self(ent) + self.W_nei(agg)), self.W_rel(rel)


class CompGCN(torch.nn.Module):
    def __init__(self, num_ent: int, num_rel: int, init_emb: torch.Tensor):
        super().__init__()
        self.ent = torch.nn.Embedding.from_pretrained(init_emb, freeze=False)
        self.rel = torch.nn.Embedding(num_rel, EMB_DIM)
        self.layers = torch.nn.ModuleList([CompGCNLayer(EMB_DIM), CompGCNLayer(EMB_DIM)])

    def encode(self, edge_index, edge_type):
        x, r = self.ent.weight, self.rel.weight
        for layer in self.layers:
            x, r = layer(x, r, edge_index, edge_type)
        return x, r

    def score(self, h, r, t):
        return (h * r * t).sum(-1)


# ════════════════════════════════════════════════════════════
# KG 加载与三元组评分
# ════════════════════════════════════════════════════════════
_kg_model_cache: dict = {}
_kg_model_cache_lock = threading.Lock()


def load_model_and_kg(law_name: str):
    mp = os.path.join(MODEL_DIR, f"{law_name}_compgcn.pt")
    kp = os.path.join(KG_DIR, f"{law_name}_kg.json")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"模型不存在: {mp}")
    if not os.path.exists(kp):
        raise FileNotFoundError(f"KG不存在:  {kp}")

    kg = KG()
    kg.load(kp)

    init_emb = torch.zeros((len(kg.entity2id), EMB_DIM), dtype=torch.float32)
    model = CompGCN(len(kg.entity2id), len(kg.rel2id), init_emb).to(DEVICE)
    ckpt = torch.load(mp, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, kg


def _build_edge_tensors(kg: KG):
    src, dst, rel = [], [], []
    for h, r, t in kg.indexed():
        src += [h, t];
        dst += [t, h];
        rel += [r, r]
    return (
        torch.tensor([src, dst], device=DEVICE),
        torch.tensor(rel, device=DEVICE),
    )


def get_triples_scored(law_name: str) -> list:
    with _kg_model_cache_lock:
        if law_name in _kg_model_cache:
            return _kg_model_cache[law_name]

    try:
        model, kg = load_model_and_kg(law_name)
    except FileNotFoundError as e:
        logging.warning(f"[KG] {e}")
        return []

    if not kg.triplets:
        logging.warning(f"[KG] {law_name} KG 为空")
        return []

    ei, et = _build_edge_tensors(kg)
    rows = []
    with torch.no_grad():
        ent, rlt = model.encode(ei, et)
        for h_s, r_s, t_s in kg.triplets:
            h = kg.entity2id.get(h_s)
            r = kg.rel2id.get(r_s)
            t = kg.entity2id.get(t_s)
            if None in (h, r, t):
                continue
            s = model.score(
                ent[h].unsqueeze(0), rlt[r].unsqueeze(0), ent[t].unsqueeze(0)
            ).item()
            rows.append({"head": h_s, "relation": r_s, "tail": t_s, "score": round(s, 4)})

    rows.sort(key=lambda x: x["score"], reverse=True)
    result = rows[:TOP_K_TRIPLES]
    with _kg_model_cache_lock:
        _kg_model_cache[law_name] = result
    return result


# ════════════════════════════════════════════════════════════
# 精确法条库
# ════════════════════════════════════════════════════════════
_law_structured: list = []
_law_structured_lock = threading.Lock()


def load_law_structured():
    global _law_structured
    with _law_structured_lock:
        if _law_structured:
            return
        p = Path(LAW_STRUCTURED_DB)
        if not p.exists():
            logging.warning(f"law_structured.json 不存在: {p}")
            return
        with open(p, encoding="utf-8") as f:
            _law_structured = json.load(f)
        print(f"[LawDB] 精确法条库 {len(_law_structured)} 条")


def retrieve_precise_chunks(statutes: list) -> list:
    if not _law_structured or not statutes:
        return []
    results, seen = [], set()
    for statute in statutes:
        law_name = statute.get("law", "").strip()
        for article in statute.get("articles", []):
            article = article.strip()
            for record in _law_structured:
                rec_law = record.get("law", "").strip()
                rec_art = record.get("article", "").strip()
                law_match = (law_name in rec_law or rec_law in law_name
                             or law_name == rec_law)
                if law_match and rec_art == article:
                    chunk = record.get("chunk", "").strip()
                    key = f"{rec_law}_{rec_art}"
                    if chunk and key not in seen:
                        results.append(chunk);
                        seen.add(key)
                    break
            if len(results) >= MAX_CHUNKS:
                break
        if len(results) >= MAX_CHUNKS:
            break
    return results


# ════════════════════════════════════════════════════════════
# DeepSeek API 调用
# ════════════════════════════════════════════════════════════
def call_deepseek(prompt: str, max_retry: int = MAX_RETRY) -> str:
    for attempt in range(max_retry):
        try:
            conn = http.client.HTTPSConnection(API_HOST)
            payload = json.dumps({
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
            })
            headers = {
                "Authorization": f"Bearer {get_api_key()}",
                "Content-Type": "application/json",
            }
            conn.request("POST", "/v1/chat/completions", payload, headers)
            raw = conn.getresponse().read().decode("utf-8")
            data = json.loads(raw)
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            else:
                logging.error(f"API error: {data}")
        except Exception as e:
            logging.error(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == max_retry - 1:
                raise
            time.sleep(2)
    raise RuntimeError("API call failed after retries")


# ════════════════════════════════════════════════════════════
# Prompt 构建 —— 四种题型
# ════════════════════════════════════════════════════════════

def build_choice_prompt(
        full_text: str,
        precise_cks: list,
        laws: list,
        triples: list,
) -> str:
    """选择题 Prompt"""
    precise_section = ""
    if precise_cks:
        precise_section = "═══ 精确法条原文 ═══\n"
        for i, chunk in enumerate(precise_cks, 1):
            precise_section += f"\n【法条 {i}】\n{chunk}\n"
        precise_section += "\n"

    laws_section = ""
    if laws and not precise_cks:
        laws_section = "═══ 参考法律（向量检索）═══\n"
        for law in laws:
            extra = ""
            if law.get("article"):
                extra += f"  条款：{law['article']}\n"
            if law.get("chapter"):
                extra += f"  章节：{law['chapter']}\n"
            laws_section += (
                f"【{law['law_name']}】相似度 {law['similarity']}\n"
                f"{extra}"
                f"相关内容：{law['matched_text'][:300]}\n\n"
            )

    kg_section = ""
    if triples and not precise_cks:
        kg_section = "═══ 知识图谱三元组（仅供参考）═══\n"
        for i, t in enumerate(triples, 1):
            kg_section += (
                f"{i}. {t['head']} --[{t['relation']}]--> {t['tail']}"
                f" (评分: {t['score']})\n"
            )
        kg_section += "\n"

    return f"""你是一位精通中国法律的资深专家。请仔细阅读以下题目，并根据法律知识进行判断。

【重要提示】
- 本题可能是单选题，也可能是多选题，请逐项分析
- 题干内容是判断的核心依据
- 精确法条原文具有最高权重
- 只输出正确选项的字母（如: A、BC、ABCD）
- 不要输出任何解释或分析过程

{precise_section}{laws_section}{kg_section}═══ 题目（核心依据）═══
{full_text}

请在仔细分析后，只输出正确选项的字母组合。
如果是单选题，输出单个字母（如：A）
如果是多选题，输出字母序列（如：ABC）"""


def build_summary_prompt(full_text: str, laws: list, triples: list) -> str:
    """摘要题 Prompt（5_1）"""
    laws_section = ""
    if laws:
        laws_section = "\n【参考法律信息】\n"
        for law in laws:
            laws_section += (
                f"• {law['law_name']} ({law['similarity']:.2%})\n"
                f"  {law['matched_text'][:200]}...\n"
            )

    kg_section = ""
    if triples:
        kg_section = "\n【相关法律概念】\n"
        for t in triples[:2]:  # 只显示top-2
            kg_section += f"• {t['head']} {t['relation']} {t['tail']}\n"

    return f"""你是一位专业的法律文案专家。请完成以下任务：

【任务】
请根据以下内容生成一段不超过400字的摘要。
摘要应当：
  1. 准确提炼核心要点
  2. 逻辑清晰、结构合理
  3. 用词准确、专业规范
  4. 可适当引用相关法律条款{laws_section}{kg_section}

【待摘要内容】
{full_text}

【要求】
- 字数：不超过400字
- 风格：学术、专业、严谨
- 格式：段落清晰

请直接给出摘要，不需要额外说明："""


def build_judgment_prompt(full_text: str, laws: list, triples: list) -> str:
    """裁判分析题 Prompt（5_2）"""
    laws_section = ""
    if laws:
        laws_section = "\n【相关法律依据】\n"
        for law in laws:
            laws_section += (
                f"• 【{law['law_name']}】相似度：{law['similarity']:.2%}\n"
                f"  条款：{law['article']} - {law['chapter']}\n"
                f"  原文：{law['matched_text'][:250]}...\n\n"
            )

    return f"""你是一位资深法官助理。请完成以下任务：

【任务】
根据以下法律案情的基本事实，生成裁判分析过程。{laws_section}

【要求】
裁判分析过程应当：
  1. 充分涵盖法院对案情中存在的争议焦点的透彻分析与回应
  2. 详细引用相关法条（格式：《法律名》第X条）
  3. 逻辑严密、论证充分
  4. 最终呈现法庭判决结果

【案情基本事实】
{full_text}

【输出格式】
一、争议焦点分析
   [分析内容]

二、法律适用
   [引用相关法条]

三、裁判理由
   [详细论证]

四、判决结果
   [判决内容]

请直接给出分析结果："""


def build_translate_prompt(full_text: str, laws: list) -> str:
    """翻译题 Prompt（5_3）"""
    context = ""
    if laws:
        context = f"\n【相关法律背景】\n{laws[0]['matched_text'][:200]}\n"

    return f"""你是一位专业的法律翻译专家。请完成以下翻译任务：

【任务】
请完成以下法律场景的翻译任务，直接给出翻译结果。{context}

【原文（中文）】
{full_text}

【要求】
  1. 准确传达原文的法律含义
  2. 使用规范的英文法律术语
  3. 保持原文的结构和逻辑
  4. 风格专业、严谨

【输出】
请直接给出英文翻译，不需要任何其他说明："""


def build_essay_prompt(full_text: str, laws: list, triples: list) -> str:
    """论述题 Prompt（5_4）"""
    laws_section = ""
    if laws:
        laws_section = "\n【参考法律资源】\n"
        for law in laws:
            laws_section += (
                f"• {law['law_name']} (相似度: {law['similarity']:.2%})\n"
                f"  {law['matched_text'][:200]}...\n"
            )

    kg_section = ""
    if triples:
        kg_section = "\n【法律概念体系】\n"
        for t in triples:
            kg_section += f"• {t['head']} - {t['relation']} - {t['tail']} (评分: {t['score']})\n"

    return f"""你是一位资深法律学者。请完成以下论述题：

【任务】
请分析以下论述题，详细阐述你的观点。{laws_section}{kg_section}

【要求】
  1. 详细阐述你的观点
  2. 充分的论据和分析
  3. 可引用法律条文和相关法律原则
  4. 清晰展示对法律问题的深刻理解和灵活应用能力
  5. 逻辑严密、表达清晰

【论述题内容】
{full_text}

【输出格式】
建议采用以下结构（可根据题目调整）：
一、问题分析
二、理论依据
三、案例应用
四、结论

请直接给出详细的论述分析："""


# ════════════════════════════════════════════════════════════
# 单条处理核心逻辑
# ════════════════════════════════════════════════════════════
def process_item(item: dict, task_name: str, question_type: str) -> dict:
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    answer = item.get("answer", "")
    full_text = (instruction + "\n" + input_text).strip()

    # Step 1: 向量检索
    laws: list = []
    try:
        laws = get_law_retriever().search(full_text)
    except Exception as e:
        logging.warning(f"[{task_name}] LawRetriever search failed: {e}")

    # Step 2: 提取法律名称
    kg_names: list = []
    for law in laws:
        norm = normalize_law_name(law["law_name"])
        if norm and norm not in kg_names:
            kg_names.append(norm)

    # Step 3: KG 三元组（非选择题也可以使用）
    all_triples: list = []
    for law_name in kg_names:
        try:
            all_triples.extend(get_triples_scored(law_name))
        except Exception as e:
            logging.warning(f"[{task_name}] KG processing {law_name}: {e}")

    all_triples.sort(key=lambda x: x["score"], reverse=True)
    top_triples = all_triples[:TOP_K_TRIPLES]

    # Step 4: 精确法条检索
    precise_cks: list = []

    if precise_cks:
        top_triples = []
        logging.info(f"[{task_name}] 精确命中法条 {len(precise_cks)} 条，清空KG")

    logging.info(
        f"[{task_name}] type={question_type} | laws={len(laws)} | "
        f"kg_names={kg_names} | triples={len(top_triples)}"
    )

    # Step 5: 根据题型选择 Prompt
    if question_type == "choice":
        prompt = build_choice_prompt(full_text, precise_cks, laws, top_triples)
    elif question_type == "summary":
        prompt = build_summary_prompt(full_text, laws, top_triples)
    elif question_type == "judgment":
        prompt = build_judgment_prompt(full_text, laws, top_triples)
    elif question_type == "translate":
        prompt = build_translate_prompt(full_text, laws)
    elif question_type == "essay":
        prompt = build_essay_prompt(full_text, laws, top_triples)
    else:
        prompt = build_choice_prompt(full_text, precise_cks, laws, top_triples)

    # Step 6: API 调用
    final = ""
    for attempt in range(MAX_RETRY):
        try:
            final = call_deepseek(prompt)
            break
        except Exception as e:
            logging.warning(f"[{task_name}] API attempt {attempt + 1}: {e}")
            if attempt == MAX_RETRY - 1:
                final = "FAILED"
            else:
                time.sleep(1)

    return {
        "input": full_text,
        "output": final,
        "answer": answer,
        "question_type": question_type,
        "retrieved_laws": [
            {"law_name": l["law_name"], "similarity": l["similarity"]}
            for l in laws
        ],
        "kg_names": kg_names,
        "kg_triples": [
            {"head": t["head"], "relation": t["relation"],
             "tail": t["tail"], "score": t["score"]}
            for t in top_triples
        ],
        "precise_chunks": precise_cks,
        "api_calls": 1,
    }


# ════════════════════════════════════════════════════════════
# 样本处理包装器
# ════════════════════════════════════════════════════════════
def process_item_with_retry(item: dict, task_name: str, question_type: str) -> dict:
    try:
        return process_item(item, task_name, question_type)
    except Exception as e:
        logging.error(f"[{task_name}] 处理异常: {e}")
        return {
            "input": item.get("input", ""),
            "output": "FAILED",
            "answer": item.get("answer", ""),
            "question_type": question_type,
            "error": str(e),
            "api_calls": 0,
        }


# ════════════════════════════════════════════════════════════
# 文件处理与主流程
# ════════════════════════════════════════════════════════════
def load_data(path: str) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass

    if not data:
        with open(path, encoding="utf-8") as f:
            try:
                raw = json.load(f)
                if isinstance(raw, list):
                    data = raw
                elif isinstance(raw, dict) and "data" in raw:
                    data = raw["data"]
            except Exception as e:
                logging.error(f"无法加载 {path}: {e}")
    return data


def run_on_file(f_path: str, question_type: str):
    task_name = os.path.basename(f_path).split(".")[0]
    data = load_data(f_path)

    if not data:
        print(f"[SKIP] {task_name}: 无数据")
        return

    print(f"\n[RUN]  {task_name} ({question_type}) | {len(data)} items")
    logging.info(f"Start {task_name}: {len(data)} items | type: {question_type}")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {
            ex.submit(process_item_with_retry, item, task_name, question_type): item
            for item in data
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc=task_name):
            try:
                results.append(fut.result())
            except Exception as e:
                logging.error(f"[{task_name}] 未捕获异常: {e}")
                results.append({
                    "input": futs[fut].get("input", ""),
                    "output": "FAILED",
                    "answer": futs[fut].get("answer", ""),
                    "question_type": question_type,
                    "error": str(e),
                    "api_calls": 0,
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"gnn_chain_{task_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    failed = sum(1 for r in results if r.get("output") == "FAILED")
    empty_kg = sum(1 for r in results if not r.get("kg_triples"))
    total_api_calls = sum(r.get("api_calls", 0) for r in results)

    print(f"[DONE] Saved -> {out_path}")
    print(f"       failed:        {failed}/{len(results)}")
    print(f"       kg_empty:      {empty_kg}/{len(results)}")
    print(f"       total_api_calls: {total_api_calls} "
          f"(avg {total_api_calls / len(results):.1f}/sample)")

# 白名单：只跑这四个
TARGET_TASKS = {"5_1", "5_2", "5_3", "5_4"}

def main():
    print("=" * 70)
    print("GNN-Chain Pipeline - 只跑 5_1 / 5_2 / 5_3 / 5_4")
    print("=" * 70)
    print("支持题型：")
    print("  • 摘要题（5_1）")
    print("  • 裁判分析（5_2）")
    print("  • 翻译题（5_3）")
    print("  • 论述题（5_4）")
    print("=" * 70)

    load_law_structured()

    print("\n[INIT] Loading LawRetriever ...")
    get_law_retriever()

    avail = list_available_kg_names()
    print(f"\n[KG]  可用 KG 文件: {len(avail)} 个")
    if avail:
        print(f"      示例: {', '.join(avail[:3])}")

    # 所有 json 文件
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))

    # ✅ 只保留 TARGET_TASKS 里的文件
    files = [
        f for f in all_files
        if os.path.basename(f).split(".")[0] in TARGET_TASKS
    ]

    if not files:
        print(f"\n[ERR] 在 {DATA_DIR} 中未找到 5_1/5_2/5_3/5_4 对应的文件")
        print(f"      当前找到的文件：{[os.path.basename(f) for f in all_files]}")
        return

    print(f"\n[DATA] 命中目标文件 {len(files)} 个：")
    for f in files:
        task_name = os.path.basename(f).split(".")[0]
        q_type    = detect_question_type(task_name)
        print(f"       {os.path.basename(f)}  -->  类型: {q_type}")

    print()

    for f in files:
        task_name = os.path.basename(f).split(".")[0]
        q_type    = detect_question_type(task_name)
        run_on_file(f, q_type)

    print("\n" + "=" * 70)
    print("5_1 / 5_2 / 5_3 / 5_4 处理完成！")
    print("=" * 70)
    logging.info("GNN-Chain 5_1~5_4 complete")


if __name__ == "__main__":
    main()