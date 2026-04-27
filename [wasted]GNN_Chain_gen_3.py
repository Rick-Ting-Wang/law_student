"""
deepseek prompt version
gnn_chain.py — GNN-Chain 两步推理脚本（增强版）
=================================================
流程:
  1. Lawformer + ChromaDB  →  检索最相关法律（本地）
  2. CompGCN               →  KG 三元组评分（本地）
  3. 精确法条检索（law_structured.json）→ 补充关键词匹配
  4. DeepSeek API Call 1   →  生成结构化推理 JSON（区分选择题 / 案例题）
  5. DeepSeek API Call 2   →  基于结构化推理给出最终答案

输出: model_output/zero_shot/GNN_Chain_2_Step/gnn_chain_{task_name}.jsonl
"""

import os, json, logging, http.client, glob, threading, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import chromadb

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
API_KEYS = [
    "sk-aZ4KJUKxqQGI0kvJBf87Aa2eE53a487e81A8D26f89CfD486",
    "sk-DJ29rXEPynvLeo1oF3B760C896Cd426cA3B08452AfF9904e",
    "sk-0ZUB8rg4mn837rP97139C01d1cF34fD6A275A651B219Ba36",
    "sk-TBnjH5MJFpVkRVnjF3E352E065F34e4fA4B395Dd16EcBe90",
    "sk-WDc1G15qcXMGqKe562A43c14Ba934807B3Bd30F8A74a3729",
    "sk-KOSjINQK6Hq6nYKq687045C77aAa4cB394Fc8e337886E49b",
    "sk-1uVFVa56Vf1UJI5EF6BfAeD331D443B49b0d1cDaB5D2E8F2",
    "sk-flmDwsSl2Nm4RK3e677cC84f904a4dE2Be0d4a11D584FcC7",
    "sk-yIP2MhrdtnzzJWU2D82bC5F313144b5cA48aCaD2126aB5A1",
    "sk-yFXgUqnTSVsr0Xhc1595245c35E347A5B7BaCf7e72Bc3cE9",
]

API_HOST   = "dpapi.cn"
MODEL_NAME = "deepseek-v3"

LAW_DB_DIR      = "./chroma_law_db"
LAW_COLLECTION  = "law_collection"
LAWFORMER_MODEL = "thunlp/Lawformer"
MAX_SEQ_LEN     = 4096
TOP_K_LAWS      = 5
TOP_K_CHUNKS    = 30

MODEL_DIR     = "./models"
KG_DIR        = "./kg_store"
EMB_DIM       = 768
TOP_K_TRIPLES = 20

DATA_DIR    = "./data"
OUT_DIR     = "./model_output/zero_shot/GNN_Chain_3"
LOG_FILE    = "[wasted]gnn_chain_running.log"

MAX_RETRY   = 3
NUM_WORKERS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 精确法条库路径
LAW_STRUCTURED_FILE = "./law_structured.json"

# 任务后缀映射（非选择题任务）
SUFFIX_MAP = {
    "5_1": "摘要:",
    "5_2": "裁判分析过程:",
    "5_3": "翻译结果:",
}
# ════════════════════════════════════════════════════════════

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── API Key 轮转 ─────────────────────────────────────────────
_tl = threading.local()
_key_lock = threading.Lock()
_key_idx  = 0

def get_api_key():
    global _key_idx
    if not hasattr(_tl, "key"):
        with _key_lock:
            _tl.key = API_KEYS[_key_idx % len(API_KEYS)]
            _key_idx += 1
    return _tl.key

# ── 题型检测 ─────────────────────────────────────────────────
_MCQ_RE = re.compile(
    r"选择题|单选|多选|下列.*?(?:正确|错误|不正确|不属于|不符合)|"
    r"(?:^|\n)[A-D][：:．.]\s",
    re.MULTILINE
)

def is_mcq(instruction: str, input_text: str) -> bool:
    """判断是否为选择题。"""
    combined = instruction + input_text
    return bool(_MCQ_RE.search(combined))

def extract_options(input_text: str) -> list[tuple[str, str]]:
    """
    从题目文本中提取选项列表，返回 [('A', '内容'), ('B', '内容'), ...]。
    """
    pattern = re.compile(r'(?:^|\n)([A-D])[：:．.]\s*(.+?)(?=\n[A-D][：:．.]|\Z)', re.S)
    return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(input_text)]

# ── 任务细分检测 ─────────────────────────────────────────────
def detect_subtask(instruction: str, input_text: str) -> str:
    """返回 subtask: summary, judgment_analysis, translation, essay, general"""
    combined = instruction + input_text
    if "摘要" in instruction and "不超过" in instruction:
        return "summary"
    if "裁判分析过程" in instruction:
        return "judgment_analysis"
    if "翻译" in instruction or "翻译为中文" in instruction:
        return "translation"
    if "论述题" in instruction or "阐述你的观点" in instruction:
        return "essay"
    return "general"

# ── Lawformer 检索器 ─────────────────────────────────────────
class LawRetriever:
    def __init__(self):
        print(f"[LawRetriever] Loading {LAWFORMER_MODEL} on {DEVICE} ...")
        self.device = str(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(LAWFORMER_MODEL)
        self.model = AutoModel.from_pretrained(LAWFORMER_MODEL).to(self.device)
        self.model.eval()
        self._lock = threading.Lock()
        db_path = Path(LAW_DB_DIR)
        if not db_path.exists():
            raise FileNotFoundError(f"ChromaDB not found: {db_path}")
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_collection(LAW_COLLECTION)
        print(f"[LawRetriever] Ready — {self.collection.count()} chunks")

    @torch.no_grad()
    def _encode(self, text):
        with self._lock:
            enc = self.tokenizer(
                [text], padding=True, truncation=True,
                max_length=MAX_SEQ_LEN, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            tok  = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(tok.size()).float()
            emb  = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(emb, p=2, dim=1).cpu().tolist()[0]

    def _aggregate(self, res):
        best = {}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            sim  = round(1 - dist, 4)
            name = meta["law_name"]
            if name not in best or sim > best[name]["similarity"]:
                best[name] = {
                    "law_name":     name,
                    "filename":     meta.get("filename", ""),
                    "similarity":   sim,
                    "chunk_index":  meta.get("chunk_index", 0),
                    "total_chunks": meta.get("total_chunks", 1),
                    "matched_text": doc,
                }
        return sorted(best.values(), key=lambda x: x["similarity"], reverse=True)[:TOP_K_LAWS]

    def search(self, query):
        if not query.strip():
            return []
        vec = self._encode(query)
        res = self.collection.query(
            query_embeddings=[vec],
            n_results=min(TOP_K_CHUNKS, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return self._aggregate(res)


_retriever = None
_retriever_lock = threading.Lock()

def get_retriever():
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = LawRetriever()
    return _retriever

# ── 精确法条检索（新增）─────────────────────────────────────
_precise_law_cache = None
_precise_law_lock = threading.Lock()

def load_precise_laws():
    global _precise_law_cache
    if _precise_law_cache is None:
        with _precise_law_lock:
            if _precise_law_cache is None:
                if os.path.exists(LAW_STRUCTURED_FILE):
                    with open(LAW_STRUCTURED_FILE, "r", encoding="utf-8") as f:
                        _precise_law_cache = json.load(f)
                else:
                    _precise_law_cache = []
    return _precise_law_cache

def precise_law_retrieve(keywords: list, top_k: int = 5) -> list:
    """
    从 law_structured.json 中根据关键词精确检索法条。
    关键词可以是法律名称、条文号、主题词等。
    """
    all_laws = load_precise_laws()
    if not all_laws:
        return []
    results = []
    for law in all_laws:
        content = law.get("chunk", "") + " " + law.get("text", "")
        score = sum(1 for kw in keywords if kw.lower() in content.lower())
        if score > 0:
            results.append((score, law))
    results.sort(key=lambda x: x[0], reverse=True)
    top = []
    for score, law in results[:top_k]:
        top.append({
            "law_name": law.get("law", ""),
            "article": law.get("article", ""),
            "text": law.get("text", ""),
            "chunk": law.get("chunk", ""),
            "similarity": round(score / max(len(keywords), 1), 4)
        })
    return top

# ── CompGCN ──────────────────────────────────────────────────
class KG:
    def __init__(self):
        self.triplets = []; self.entity2id = {}; self.rel2id = {}

    def load(self, path):
        d = json.load(open(path, encoding="utf-8"))
        self.name = d["name"]; self.triplets = d["triplets"]
        self.entity2id = d["entity2id"]; self.rel2id = d["rel2id"]

    def indexed(self):
        return [(self.entity2id[h], self.rel2id[r], self.entity2id[t])
                for h, r, t in self.triplets
                if h in self.entity2id and r in self.rel2id and t in self.entity2id]


class _GCNLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Ws = torch.nn.Linear(dim, dim)
        self.Wn = torch.nn.Linear(dim, dim)
        self.Wr = torch.nn.Linear(dim, dim)
        self.act = torch.nn.ReLU()
    def forward(self, e, r, idx, et):
        s, d = idx
        msg  = e[s] * torch.sigmoid(r[et])
        agg  = torch.zeros_like(e); agg.index_add_(0, d, msg)
        return self.act(self.Ws(e) + self.Wn(agg)), self.Wr(r)


class CompGCN(torch.nn.Module):
    def __init__(self, ne, nr):
        super().__init__()
        self.ent    = torch.nn.Embedding(ne, EMB_DIM)
        self.rel    = torch.nn.Embedding(nr, EMB_DIM)
        self.layers = torch.nn.ModuleList([_GCNLayer(EMB_DIM), _GCNLayer(EMB_DIM)])
    def encode(self, idx, et):
        e, r = self.ent.weight, self.rel.weight
        for l in self.layers: e, r = l(e, r, idx, et)
        return e, r
    def score(self, h, r, t):
        return (h * r * t).sum(-1)


_kg_cache = {}
_kg_lock  = threading.Lock()

def get_triples_scored(law_name):
    with _kg_lock:
        if law_name in _kg_cache:
            return _kg_cache[law_name]
    mp = os.path.join(MODEL_DIR, f"{law_name}_compgcn.pt")
    kp = os.path.join(KG_DIR,    f"{law_name}_kg.json")
    if not os.path.exists(mp) or not os.path.exists(kp):
        return []
    kg = KG(); kg.load(kp)
    model = CompGCN(len(kg.entity2id), len(kg.rel2id)).to(DEVICE)
    ckpt  = torch.load(mp, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    idx = kg.indexed()
    if not idx: return []
    src, dst, rel = [], [], []
    for h, r, t in idx:
        src += [h, t]; dst += [t, h]; rel += [r, r]
    ei = torch.tensor([src, dst], device=DEVICE)
    et = torch.tensor(rel, device=DEVICE)
    with torch.no_grad():
        ent, rlt = model.encode(ei, et)
    rows = []
    for h, r, t in kg.triplets:
        if h not in kg.entity2id or r not in kg.rel2id or t not in kg.entity2id: continue
        s = model.score(
            ent[kg.entity2id[h]].unsqueeze(0),
            rlt[kg.rel2id[r]].unsqueeze(0),
            ent[kg.entity2id[t]].unsqueeze(0)
        ).item()
        rows.append({"head": h, "relation": r, "tail": t, "score": round(s, 4)})
    rows.sort(key=lambda x: x["score"], reverse=True)
    result = rows[:TOP_K_TRIPLES]
    with _kg_lock:
        _kg_cache[law_name] = result
    return result

# ── DeepSeek API ─────────────────────────────────────────────
def call_deepseek(prompt: str) -> str:
    conn = http.client.HTTPSConnection(API_HOST)
    payload = json.dumps({"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]})
    headers = {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}
    conn.request("POST", "/v1/chat/completions", payload, headers)
    raw  = conn.getresponse().read().decode("utf-8")
    data = json.loads(raw)
    if "choices" not in data:
        raise ValueError(f"API error: {data}")
    return data["choices"][0]["message"]["content"]

# ════════════════════════════════════════════════════════════
# Prompt 构建（区分选择题 / 非选择题，强制 JSON 输出）
# ════════════════════════════════════════════════════════════

def _fmt_laws(laws):
    return "\n".join(
        f"{i+1}. 【{l['law_name']}】相似度 {l['similarity']}\n"
        f"   {l['matched_text'][:250]}"
        for i, l in enumerate(laws)
    ) or "  （无检索结果）"

def _fmt_triples(triples):
    return "\n".join(
        f"  {t['head']} --[{t['relation']}]--> {t['tail']}  (score:{t['score']})"
        for t in triples[:30]
    ) or "  （无三元组数据）"

def _fmt_precise_laws(laws):
    if not laws:
        return "  （无精确检索结果）"
    return "\n".join(
        f"{i+1}. 【{l['law_name']}】第{l.get('article','')}条 相似度 {l['similarity']}\n"
        f"   {l['chunk'][:300]}"
        for i, l in enumerate(laws)
    )

# ── 选择题推理 Prompt（JSON 输出）────────────────────────────
def build_mcq_reasoning_prompt(question: str, options: list, laws: list, triples: list, precise_laws: list) -> str:
    opts_text = "\n".join(f"  {k}：{v}" for k, v in options) if options else question
    return f"""你是专业法律分析师。请依据以下检索到的法律条文和知识图谱三元组，
对下列选择题的**每一个选项**进行逐一法律分析，判断其正误，并给出依据。
最后输出一个 JSON 对象，格式如下：
{{
  "option_analysis": [
    {{"option": "A", "correct": true/false, "reason": "分析依据", "law_cited": "法条编号"}},
    ...
  ],
  "final_answer": "正确选项字母，如 A 或 BC 或 ABCD"
}}

注意：
- final_answer 中多选用字母连写，不要有空格或其他字符。
- 如果无法确定，final_answer 可设为 "UNKNOWN"。

═══ 题目 ═══
{question}

═══ 选项 ═══
{opts_text}

═══ 检索到的相关法律（Lawformer + ChromaDB）═══
{_fmt_laws(laws)}

═══ 精确检索法条（law_structured.json）═══
{_fmt_precise_laws(precise_laws)}

═══ 知识图谱三元组（CompGCN 评分）═══
{_fmt_triples(triples)}

请输出 JSON："""

# ── 非选择题推理 Prompt（JSON 输出，按子任务定制）────────────
def build_case_reasoning_prompt(case_text: str, subtask: str, laws: list, triples: list, precise_laws: list) -> str:
    if subtask == "summary":
        instructions = """请提取以下关键信息并输出 JSON：
{
  "key_facts": "主体、事件、法律争议点、判决结果",
  "legal_basis": "引用的法条编号及核心内容"
}"""
    elif subtask == "judgment_analysis":
        instructions = """请模拟法官思维，输出 JSON：
{
  "disputed_issues": ["争议焦点1", "争议焦点2"],
  "applicable_laws": ["法律名称 第X条", ...],
  "reasoning_structure": "对各方主张的回应逻辑简述",
  "proposed_judgment": "初步判决结果"
}"""
    elif subtask == "translation":
        instructions = """请输出 JSON：
{
  "explanation": "术语含义解释",
  "translation": "中文翻译结果"
}"""
    elif subtask == "essay":
        instructions = """请输出 JSON：
{
  "outline": ["观点1", "观点2", ...],
  "key_principles": ["坚持党的领导", ...],
  "argument_structure": "论述框架"
}"""
    else:
        instructions = """请输出结构化法律推理图谱 JSON，包含以下节点：
{
  "facts": "核心事实要素",
  "legal_relations": "关键实体关系路径",
  "applicable_laws": ["法条编号及内容"],
  "elements_check": "构成要件检验",
  "preliminary_conclusion": "初步法律定性"
}"""

    return f"""你是专业法律分析师。请根据以下检索信息，为案件生成结构化推理。{instructions}

═══ 案情及任务 ═══
{subtask.upper()} 任务
{case_text}

═══ 检索到的相关法律（Lawformer + ChromaDB）═══
{_fmt_laws(laws)}

═══ 精确检索法条（law_structured.json）═══
{_fmt_precise_laws(precise_laws)}

═══ 知识图谱三元组（CompGCN 评分）═══
{_fmt_triples(triples)}

请输出 JSON："""

# ── 第二次调用 Prompt 构建 ───────────────────────────────────
def build_mcq_answer_prompt(question: str, reasoning_json: dict) -> str:
    final_ans = reasoning_json.get("final_answer", "UNKNOWN")
    # 直接返回 final_answer，但为了防止模型不听话，仍然用简短提示
    return f"""根据以下分析，正确选项为 {final_ans}。请直接输出：{final_ans}"""

def build_case_answer_prompt(case_text: str, subtask: str, reasoning_json: dict) -> str:
    if subtask == "summary":
        return f"""请根据以下结构化信息，生成一段不超过400字的摘要。
案情：{case_text}
结构化信息：{json.dumps(reasoning_json, ensure_ascii=False, indent=2)}
摘要："""
    elif subtask == "judgment_analysis":
        return f"""请根据以下信息，生成完整的裁判分析过程（需引用具体法条全文，并得出判决结果）。
案情：{case_text}
结构化信息：{json.dumps(reasoning_json, ensure_ascii=False, indent=2)}
裁判分析过程："""
    elif subtask == "translation":
        trans = reasoning_json.get("translation", "")
        if trans:
            return f"翻译结果：{trans}"
        else:
            return f"请将以下法律术语翻译为中文，只输出翻译结果：\n{case_text}"
    elif subtask == "essay":
        return f"""请根据以下大纲和原则，撰写一篇不少于600字的论述文章。
题目：{case_text}
结构化信息：{json.dumps(reasoning_json, ensure_ascii=False, indent=2)}
文章："""
    else:
        return f"""请根据以下推理图谱，给出最终答案。
案情：{case_text}
推理图谱：{json.dumps(reasoning_json, ensure_ascii=False, indent=2)}
答案："""

# ════════════════════════════════════════════════════════════
# 单条处理
# ════════════════════════════════════════════════════════════
def process_item(item: dict, task_name: str) -> dict:
    instruction = item.get("instruction", "")
    input_text  = item.get("input", "")
    answer      = item.get("answer", "")
    full_text   = instruction + input_text

    # 本地检索（Lawformer）
    laws, all_triples = [], []
    try:
        laws = get_retriever().search(full_text)
    except Exception as e:
        logging.warning(f"[{task_name}] Law retrieval failed: {e}")

    # 精确法条检索（新增）
    keywords = re.findall(r'[《]([^》]+)[》]', full_text)  # 提取书名号内法律名
    keywords += re.findall(r'第\s*(\d+)\s*条', full_text)   # 提取条文号
    if not keywords:
        # 若没有明确法律名，则取前200字符中的中文词汇
        keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', full_text[:200])[:10]
    precise_laws = precise_law_retrieve(keywords, top_k=5)

    # 知识图谱
    for law in laws:
        try:
            ts = get_triples_scored(law["law_name"])
            for t in ts:
                t["_law"] = law["law_name"]
            all_triples.extend(ts)
        except Exception as e:
            logging.warning(f"[{task_name}] KG failed for {law['law_name']}: {e}")
    all_triples.sort(key=lambda x: x["score"], reverse=True)
    seen, deduped = set(), []
    for t in all_triples:
        key = (t["head"], t["relation"], t["tail"])
        if key not in seen:
            seen.add(key); deduped.append(t)
        if len(deduped) >= 40: break

    # 题型判断
    mcq = is_mcq(instruction, input_text)
    subtask = detect_subtask(instruction, input_text) if not mcq else "mcq"
    options = extract_options(input_text) if mcq else []

    # 第一次 API 调用：生成结构化推理
    thought_graph_raw = ""
    thought_graph_json = None
    for attempt in range(MAX_RETRY):
        try:
            if mcq:
                p1 = build_mcq_reasoning_prompt(full_text, options, laws, deduped, precise_laws)
            else:
                p1 = build_case_reasoning_prompt(full_text, subtask, laws, deduped, precise_laws)
            response = call_deepseek(p1)
            # 提取 JSON
            clean = re.sub(r'```json\s*|\s*```', '', response.strip())
            start = clean.find('{')
            end = clean.rfind('}')
            if start != -1 and end != -1:
                clean = clean[start:end+1]
                thought_graph_json = json.loads(clean)
            else:
                raise ValueError("No JSON object found in response")
            thought_graph_raw = response
            break
        except Exception as e:
            logging.warning(f"[{task_name}] Thought graph attempt {attempt+1}: {e}")
            if attempt == MAX_RETRY - 1:
                thought_graph_raw = "FAILED"
                thought_graph_json = None
            else:
                time.sleep(1)

    if thought_graph_raw == "FAILED":
        return _failed(full_text, answer, laws, deduped)

    # 第二次 API 调用：生成最终答案
    final = ""
    for attempt in range(MAX_RETRY):
        try:
            if mcq:
                p2 = build_mcq_answer_prompt(full_text, thought_graph_json if thought_graph_json else {})
            else:
                p2 = build_case_answer_prompt(full_text, subtask, thought_graph_json if thought_graph_json else {})
            final = call_deepseek(p2)
            break
        except Exception as e:
            logging.warning(f"[{task_name}] Answer attempt {attempt+1}: {e}")
            if attempt == MAX_RETRY - 1:
                final = "FAILED"
            else:
                time.sleep(1)

    return {
        "input":         full_text,
        "output":        final,
        "answer":        answer,
        "thought_graph": thought_graph_raw,
        "is_mcq":        mcq,
        "subtask":       subtask,
        "retrieved_laws": [
            {"law_name": l["law_name"], "similarity": l["similarity"],
             "matched_text": l["matched_text"][:200]}
            for l in laws
        ],
        "precise_laws":  precise_laws,
        "kg_triples": [
            {"head": t["head"], "relation": t["relation"],
             "tail": t["tail"],  "score":    t["score"], "law": t.get("_law", "")}
            for t in deduped[:20]
        ],
    }

def _failed(text, answer, laws, triples):
    return {
        "input": text, "output": "FAILED", "answer": answer,
        "thought_graph": "FAILED", "is_mcq": False,
        "retrieved_laws": [], "kg_triples": [],
    }

# ════════════════════════════════════════════════════════════
# 文件 / 主流程
# ════════════════════════════════════════════════════════════
def load_data(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: data.append(json.loads(line))
                except: pass
    if not data:
        data = json.load(open(path, encoding="utf-8"))
    return data

def run_on_file(f_path):
    task_name = os.path.basename(f_path).split(".")[0]
    data      = load_data(f_path)
    mcq_count = sum(1 for d in data if is_mcq(d.get("instruction",""), d.get("input","")))
    print(f"\n🚀  {task_name} | {len(data)} items | MCQ: {mcq_count} | Case: {len(data)-mcq_count}")
    logging.info(f"Start {task_name}: {len(data)} items, MCQ={mcq_count}")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(process_item, item, task_name): item for item in data}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=task_name):
            try:
                results.append(fut.result())
            except Exception as e:
                logging.error(f"[{task_name}] Unhandled: {e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"gnn_chain_{task_name}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    failed = sum(1 for r in results if r["output"] == "FAILED")
    print(f"✅  Saved → {out}  (failed: {failed}/{len(results)})")
    if failed:
        logging.warning(f"[{task_name}] {failed}/{len(results)} failed")

def main():
    print("=" * 60)
    print("GNN-Chain Pipeline (Enhanced with precise law retrieval)")
    print("  Step 1  Lawformer law retrieval        [local]")
    print("  Step 2  CompGCN KG triple scoring      [local]")
    print("  Step 3  Precise law retrieve from law_structured.json [local]")
    print("  Step 4  DeepSeek Call 1  → Structured reasoning (JSON)")
    print("          · MCQ  : per-option analysis + final_answer")
    print("          · Case : task-specific JSON (summary/judgment/translation/essay)")
    print("  Step 5  DeepSeek Call 2  → Final answer")
    print("=" * 60)

    print("\nInitializing Lawformer ...")
    get_retriever()
    print("Loading precise law database ...")
    load_precise_laws()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not files:
        print(f"❌  No .json files in {DATA_DIR}")
        return
    print(f"\nFound {len(files)} task file(s)\n")
    for f in files:
        run_on_file(f)
    print("\n🎉  ALL DONE")
    logging.info("GNN-Chain complete")

if __name__ == "__main__":
    main()