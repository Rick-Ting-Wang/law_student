"""
claude prompt version
gnn_chain.py — GNN-Chain 两步推理脚本（重构版）
=================================================
改动要点（对比原版）：
  1. Stage-1 prompt 改为结构化 JSON 输出，返回 statutes + task_type
  2. 新增 law_structured.json 精确法条检索（精确匹配 law + article）
     作为 ChromaDB 模糊检索的增强层
  3. Stage-2 将精确法条文本注入 prompt，不再只依赖相似度片段
  4. 每种 task_type 使用专属 prompt（choice/ner/relation/summary/judgment/defense/translation）
  5. 选择题明确提示"可能是多选题"
  6. JSON 解析失败自动重试 + 降级策略

流程:
  1. Lawformer + ChromaDB  →  检索最相关法律（本地）
  2. CompGCN               →  KG 三元组评分（本地）
  3. DeepSeek Call 1       →  输出结构化 JSON（statutes + task_type）
  4. 精确检索 law_structured.json 拿到法条原文
  5. DeepSeek Call 2       →  基于精确法条给出最终答案

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

LAW_DB_DIR        = "./chroma_law_db"
LAW_COLLECTION    = "law_collection"
LAW_STRUCTURED_DB = "./law_structured.json"   # ← 新增：精确法条库
LAWFORMER_MODEL   = "thunlp/Lawformer"
MAX_SEQ_LEN       = 4096
TOP_K_LAWS        = 5
TOP_K_CHUNKS      = 30

MODEL_DIR     = "./models"
KG_DIR        = "./kg_store"
EMB_DIM       = 768
TOP_K_TRIPLES = 20

DATA_DIR    = "./data"
OUT_DIR     = "./model_output/zero_shot/GNN_Chain_2"
LOG_FILE    = "[wasted]gnn_chain_running.log"

MAX_RETRY   = 3
NUM_WORKERS = 10
MAX_CHUNKS  = 8      # 注入 Stage-2 的最大精确法条数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════
# 基础设施
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

# ────────────────────────────────────────────────────────────
# 精确法条库（law_structured.json）
# ────────────────────────────────────────────────────────────

_law_structured: list[dict] = []
_law_structured_lock = threading.Lock()

def load_law_structured():
    """加载 law_structured.json，只需一次。"""
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
        logging.info(f"精确法条库加载完成: {len(_law_structured)} 条")
        print(f"[LawDB] 精确法条库 {len(_law_structured)} 条")


def retrieve_precise_chunks(statutes: list[dict]) -> list[str]:
    """
    根据 Stage-1 返回的 statutes，在 law_structured.json 中精确检索法条原文。
    使用双向模糊匹配（in）处理法律名简称问题。
    返回 chunk 字段列表，最多 MAX_CHUNKS 条。
    """
    if not _law_structured or not statutes:
        return []

    results: list[str] = []
    seen: set[str] = set()

    for statute in statutes:
        law_name: str = statute.get("law", "").strip()
        articles: list[str] = statute.get("articles", [])

        for article in articles:
            article = article.strip()
            for record in _law_structured:
                rec_law  = record.get("law", "").strip()
                rec_art  = record.get("article", "").strip()

                # 模糊匹配法律名（处理简称）
                law_match = (
                    law_name in rec_law
                    or rec_law in law_name
                    or law_name == rec_law
                )
                # 条文编号精确匹配
                art_match = rec_art == article

                if law_match and art_match:
                    chunk = record.get("chunk", "").strip()
                    key   = f"{rec_law}_{rec_art}"
                    if chunk and key not in seen:
                        results.append(chunk)
                        seen.add(key)
                    break   # 同一条文只取一次

            if len(results) >= MAX_CHUNKS:
                break
        if len(results) >= MAX_CHUNKS:
            break

    return results[:MAX_CHUNKS]


# ════════════════════════════════════════════════════════════
# Lawformer 检索器（保持原逻辑）
# ════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════
# CompGCN（保持原逻辑）
# ════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════
# DeepSeek API
# ════════════════════════════════════════════════════════════

def call_deepseek(prompt: str) -> str:
    conn = http.client.HTTPSConnection(API_HOST)
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    })
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json"
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    raw  = conn.getresponse().read().decode("utf-8")
    data = json.loads(raw)
    if "choices" not in data:
        raise ValueError(f"API error: {data}")
    return data["choices"][0]["message"]["content"].strip()


# ════════════════════════════════════════════════════════════
# Stage-1 Prompt：统一输出结构化 JSON
# ════════════════════════════════════════════════════════════

STAGE1_SYSTEM_BLOCK = """你是一名中国法律专家，精通刑法、民法典、劳动法、诉讼法及各类行政法规。

任务：阅读题目内容，通过法律知识图谱推理，找出所有需要引用的法条，输出结构化 JSON。

只输出 JSON，不要任何 markdown 代码块、解释或前缀。格式严格如下：
{
  "reasoning": "简要推理过程（100字以内）",
  "statutes": [
    {"law": "中华人民共和国刑法", "articles": ["第二百六十四条", "第二十五条"]},
    {"law": "中华人民共和国民法典", "articles": ["第六百七十五条"]}
  ],
  "task_type": "choice|ner|relation|summary|judgment|translation|defense"
}

task_type 说明（只能填其中一个值）：
- choice      选择题（单选或多选，需判断是否多选）
- ner         命名实体识别
- relation    关系三元组抽取
- summary     摘要生成
- judgment    裁判分析 / 刑期预测 / 案由预测
- translation 法律术语翻译
- defense     辩护观点匹配

注意：
- 法律名称使用全称，如"中华人民共和国刑法"而非"刑法"
- 条文编号使用中文数字全称，如"第二百六十四条"
- 如涉及共同犯罪/量刑，同时列出总则条文（第二十五条等）
- 如无法确定具体条文，statutes 留空数组 []"""


def build_stage1_prompt(laws_ctx: str, triples_ctx: str, full_text: str) -> str:
    return f"""{STAGE1_SYSTEM_BLOCK}

═══ Lawformer 检索到的相关法律 ═══
{laws_ctx}

═══ CompGCN 知识图谱三元组 ═══
{triples_ctx}

═══ 题目内容 ═══
{full_text}"""


def parse_stage1_json(raw: str) -> dict:
    """从 Stage-1 输出中提取 JSON，处理 markdown 包裹、多余前缀等。"""
    # 去 markdown 代码块
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # 直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 提取第一个 {...}
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析 Stage-1 JSON: {raw[:300]}")


# ════════════════════════════════════════════════════════════
# Stage-2 Prompt：按 task_type 专属处理
# ════════════════════════════════════════════════════════════

def _fmt_laws(laws: list) -> str:
    return "\n".join(
        f"{i+1}. 【{l['law_name']}】相似度 {l['similarity']}\n"
        f"   {l['matched_text'][:200]}"
        for i, l in enumerate(laws)
    ) or "  （无检索结果）"

def _fmt_triples(triples: list) -> str:
    return "\n".join(
        f"  {t['head']} --[{t['relation']}]--> {t['tail']}  (score:{t['score']})"
        for t in triples[:20]
    ) or "  （无三元组数据）"

def _fmt_precise(chunks: list) -> str:
    if not chunks:
        return "  （未检索到精确法条，请基于通用法律知识作答）"
    return "\n\n---\n\n".join(chunks)


# ── choice ────────────────────────────────────────────────
def build_choice_prompt(full_text: str, precise: str, laws: list, triples: list) -> str:
    return f"""你是专业法律专家。请根据以下精确法条原文和知识图谱信息，回答选择题。

【重要提示】本题可能是单选题，也可能是多选题，请仔细判断每个选项，不要遗漏正确答案。

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

═══ Lawformer 相关法律（辅助参考）═══
{_fmt_laws(laws)}

═══ 知识图谱三元组 ═══
{_fmt_triples(triples)}

═══ 题目 ═══
{full_text}

分析步骤：
1. 逐一审查每个选项的法律含义
2. 对照精确法条判断每个选项正误
3. 确认是否有多个正确选项

只输出正确选项的字母，多选用连续字母（如 A、BC、ABCD），不要解释。"""


# ── ner ───────────────────────────────────────────────────
def build_ner_prompt(full_text: str, precise: str, laws: list, triples: list) -> str:
    return f"""你是法律信息抽取专家。请从文本中抽取命名实体。

═══ 精确法条原文（辅助理解实体类型定义）═══
{_fmt_precise(precise)}

═══ 知识图谱三元组 ═══
{_fmt_triples(triples)}

═══ 题目 ═══
{full_text}

要求：
- 严格按照题目给定的实体类型列表进行分类
- 每个实体写成 (类型:实体值)，多个实体逗号分隔
- 只抽取文中实际出现的实体，不推断补充
- 只输出选项字母，不要重新构造实体列表"""


# ── relation ──────────────────────────────────────────────
def build_relation_prompt(full_text: str, precise: str, laws: list, triples: list) -> str:
    return f"""你是法律关系抽取专家。请从文本中抽取关系三元组。

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

═══ 知识图谱三元组（参考关系模式）═══
{_fmt_triples(triples)}

═══ 题目 ═══
{full_text}

要求：
- 关系类型严格限于题目给定的列表
- 三元组格式：(主体, 关系, 客体)
- 主体和客体必须是文中出现的具名实体
- 只输出选项字母"""


# ── summary ───────────────────────────────────────────────
def build_summary_prompt(full_text: str, precise: str, laws: list) -> str:
    return f"""你是法律文书编辑。请生成不超过400字的简明摘要。

═══ 相关法条（辅助理解法律术语）═══
{_fmt_precise(precise)}

═══ 参考法律（Lawformer）═══
{_fmt_laws(laws)}

═══ 案情原文 ═══
{full_text}

要求：
- 保留核心当事人、时间、争议焦点和裁判结果
- 语言简洁，使用第三人称
- 不得添加原文没有的信息
- 法律术语准确（如"商标权"不写成"品牌权"）

摘要:"""


# ── judgment（案由/刑期预测/裁判分析）──────────────────────
def build_judgment_prompt(full_text: str, precise: str, laws: list, triples: list,
                          suffix: str) -> str:
    return f"""你是资深法律专家。请根据精确法条和案情事实给出准确判断。

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

═══ Lawformer 相关法律 ═══
{_fmt_laws(laws)}

═══ 知识图谱三元组 ═══
{_fmt_triples(triples)}

═══ 案情及问题 ═══
{full_text}

量刑/案由参考因素：
- 犯罪数额与法定量刑幅度
- 是否有从轻/从重情节（自首、坦白、累犯等）
- 是否为初犯、是否退赃
- 参照检察院量刑建议

对于选择题：只输出选项字母，不要解释。
对于问答题：语言专业准确，引用具体法条编号。

{suffix}"""


# ── defense ───────────────────────────────────────────────
def build_defense_prompt(full_text: str, precise: str, laws: list) -> str:
    return f"""你是法律辩护专家。请阅读诉方观点，从选项中找出与之直接对应的辩方论点。

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

═══ 参考法律 ═══
{_fmt_laws(laws)}

═══ 题目 ═══
{full_text}

匹配标准：
- 辩方观点应针对诉方具体主张作出回应
- 不要选择无关的程序性陈述或泛泛表态
- 优先选择承认部分事实但争辩法律责任的选项

只输出选项字母。"""


# ── translation ───────────────────────────────────────────
def build_translation_prompt(full_text: str) -> str:
    return f"""你是精通中英文法律术语的专业译者。

任务：将给定法律术语/原则准确翻译为中文。

要求：
- 使用中国法律实践中的通用译法
- 如有多种译法，给出最常见的一种
- 只输出翻译结果，不要解释

{full_text}

翻译结果:"""


# ── 路由函数 ──────────────────────────────────────────────
SUFFIX_MAP = {
    "5_1": "",        # summary 任务不需要 suffix
    "5_2": "裁判分析过程:",
    "5_3": "",        # translation 独立处理
}

def build_stage2_prompt(
    task_type: str,
    full_text: str,
    precise_chunks: list,
    laws: list,
    triples: list,
    task_name: str,
) -> str:
    """根据 task_type 选择对应的 Stage-2 prompt 构建函数。"""
    if task_type == "choice":
        return build_choice_prompt(full_text, precise_chunks, laws, triples)
    elif task_type == "ner":
        return build_ner_prompt(full_text, precise_chunks, laws, triples)
    elif task_type == "relation":
        return build_relation_prompt(full_text, precise_chunks, laws, triples)
    elif task_type == "summary":
        return build_summary_prompt(full_text, precise_chunks, laws)
    elif task_type == "judgment":
        suffix = SUFFIX_MAP.get(task_name, "答案:")
        return build_judgment_prompt(full_text, precise_chunks, laws, triples, suffix)
    elif task_type == "defense":
        return build_defense_prompt(full_text, precise_chunks, laws)
    elif task_type == "translation":
        return build_translation_prompt(full_text)
    else:
        # 未知类型降级为 choice
        return build_choice_prompt(full_text, precise_chunks, laws, triples)


# ════════════════════════════════════════════════════════════
# 单条处理（核心流程）
# ════════════════════════════════════════════════════════════

def process_item(item: dict, task_name: str) -> dict:
    instruction = item.get("instruction", "")
    input_text  = item.get("input", "")
    answer      = item.get("answer", "")
    full_text   = (instruction + "\n" + input_text).strip()

    # ── Step 1: Lawformer 检索 ──────────────────────────────
    laws: list = []
    try:
        laws = get_retriever().search(full_text)
    except Exception as e:
        logging.warning(f"[{task_name}] Law retrieval failed: {e}")

    # ── Step 2: CompGCN 三元组评分 ──────────────────────────
    all_triples: list = []
    for law in laws:
        try:
            ts = get_triples_scored(law["law_name"])
            for t in ts:
                t["_law"] = law["law_name"]
            all_triples.extend(ts)
        except Exception as e:
            logging.warning(f"[{task_name}] KG failed for {law['law_name']}: {e}")

    all_triples.sort(key=lambda x: x["score"], reverse=True)
    seen_triples, deduped = set(), []
    for t in all_triples:
        key = (t["head"], t["relation"], t["tail"])
        if key not in seen_triples:
            seen_triples.add(key); deduped.append(t)
        if len(deduped) >= 40:
            break

    # ── Step 3: Stage-1 API 调用（输出结构化 JSON）──────────
    laws_ctx    = _fmt_laws(laws)
    triples_ctx = _fmt_triples(deduped)
    p1 = build_stage1_prompt(laws_ctx, triples_ctx, full_text)

    stage1_raw  = ""
    stage1_data = {}
    for attempt in range(MAX_RETRY):
        try:
            stage1_raw  = call_deepseek(p1)
            stage1_data = parse_stage1_json(stage1_raw)
            break
        except Exception as e:
            logging.warning(f"[{task_name}] Stage-1 attempt {attempt+1}: {e}")
            if attempt == MAX_RETRY - 1:
                # 降级：用 ChromaDB 结果中的法律名填充 statutes，task_type 猜测
                stage1_data = {
                    "reasoning": "Stage-1 failed, fallback",
                    "statutes": [
                        {"law": l["law_name"], "articles": []}
                        for l in laws[:3]
                    ],
                    "task_type": _guess_task_type(instruction, input_text),
                }
            else:
                time.sleep(1)

    # ── Step 4: 精确法条检索 ────────────────────────────────
    statutes     = stage1_data.get("statutes", [])
    task_type    = stage1_data.get("task_type", "choice")
    precise_cks  = retrieve_precise_chunks(statutes)

    logging.info(
        f"[{task_name}] task_type={task_type} | statutes={len(statutes)} "
        f"| precise_chunks={len(precise_cks)}"
    )

    # ── Step 5: Stage-2 API 调用（精确法条注入 + 专属 prompt）
    p2    = build_stage2_prompt(task_type, full_text, precise_cks, laws, deduped, task_name)
    final = ""
    for attempt in range(MAX_RETRY):
        try:
            final = call_deepseek(p2)
            break
        except Exception as e:
            logging.warning(f"[{task_name}] Stage-2 attempt {attempt+1}: {e}")
            if attempt == MAX_RETRY - 1:
                final = "FAILED"
            else:
                time.sleep(1)

    return {
        "input":            full_text,
        "output":           final,
        "answer":           answer,
        "task_type":        task_type,
        "stage1_reasoning": stage1_data.get("reasoning", ""),
        "stage1_statutes":  statutes,
        "precise_chunks":   precise_cks,
        "retrieved_laws": [
            {"law_name": l["law_name"], "similarity": l["similarity"],
             "matched_text": l["matched_text"][:200]}
            for l in laws
        ],
        "kg_triples": [
            {"head": t["head"], "relation": t["relation"],
             "tail": t["tail"], "score": t["score"], "law": t.get("_law", "")}
            for t in deduped[:20]
        ],
    }


# ── task_type 降级猜测 ────────────────────────────────────
_MCQ_RE = re.compile(
    r"选择题|单选|多选|下列.*?(?:正确|错误|不正确|不属于|不符合)|"
    r"(?:^|\n)[A-D][：:．.]\s",
    re.MULTILINE
)

def _guess_task_type(instruction: str, input_text: str) -> str:
    combined = instruction + input_text
    if _MCQ_RE.search(combined):
        return "choice"
    if "实体" in combined or "命名" in combined:
        return "ner"
    if "三元组" in combined or "关系" in combined:
        return "relation"
    if "摘要" in combined:
        return "summary"
    if "翻译" in combined:
        return "translation"
    if "辩方" in combined or "辩护" in combined:
        return "defense"
    return "judgment"


# ════════════════════════════════════════════════════════════
# 文件 / 主流程
# ════════════════════════════════════════════════════════════

def load_data(path: str) -> list[dict]:
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
        raw = json.load(open(path, encoding="utf-8"))
        # 兼容 grouped 格式 [{"filename":..., "first_item":{...}}, ...]
        if raw and isinstance(raw[0], dict) and "first_item" in raw[0]:
            data = [item["first_item"] for item in raw if "first_item" in item]
        else:
            data = raw
    return data


def run_on_file(f_path: str):
    task_name = os.path.basename(f_path).split(".")[0]
    data      = load_data(f_path)

    choice_cnt = sum(
        1 for d in data
        if _guess_task_type(d.get("instruction",""), d.get("input","")) == "choice"
    )
    print(f"\n🚀  {task_name} | {len(data)} items | ~choice: {choice_cnt}")
    logging.info(f"Start {task_name}: {len(data)} items")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(process_item, item, task_name): item for item in data}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=task_name):
            try:
                results.append(fut.result())
            except Exception as e:
                logging.error(f"[{task_name}] Unhandled: {e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"gnn_chain_{task_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    failed = sum(1 for r in results if r.get("output") == "FAILED")
    no_precise = sum(1 for r in results if not r.get("precise_chunks"))
    print(
        f"✅  Saved → {out_path}\n"
        f"    failed: {failed}/{len(results)}\n"
        f"    无精确法条(降级到Chroma): {no_precise}/{len(results)}"
    )
    logging.info(f"[{task_name}] done. failed={failed}, no_precise={no_precise}")


def main():
    print("=" * 65)
    print("GNN-Chain Pipeline  (精确法条检索 + task-aware prompt)")
    print("  Step 1  Lawformer + ChromaDB  →  语义检索相关法律")
    print("  Step 2  CompGCN               →  KG 三元组评分")
    print("  Step 3  DeepSeek Call 1       →  结构化 JSON（statutes + task_type）")
    print("  Step 4  law_structured.json   →  精确法条原文检索")
    print("  Step 5  DeepSeek Call 2       →  task-aware prompt → 最终答案")
    print("=" * 65)

    # 加载精确法条库
    load_law_structured()

    # 初始化 Lawformer
    print("\nInitializing Lawformer ...")
    get_retriever()

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