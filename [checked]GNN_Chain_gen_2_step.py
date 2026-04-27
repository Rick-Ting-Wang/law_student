"""
gnn_chain.py — GNN-Chain 两步推理脚本（速度优化版）
===================================================
优化点：
  1. 知识图谱匹配：Lawformer 检索法律名称 → 直接加载对应 CompGCN
  2. 样本级自动重试 → 关闭（SAMPLE_MAX_RETRY = 1），FAILED 直接返回
  3. Stage-2 明确强调题干重要性，推理链仅供参考，KG 为空时不显示
  4. API 内部重试次数降低为 3 次，提升响应速度
  5. 增加 Stage-1 推理链注入 Stage-2 Prompt
"""

import os, json, logging, http.client, glob, threading, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
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
API_HOST   = "dpapi.cn"
MODEL_NAME = "deepseek-v3"

# 法律检索路径（Lawformer）
CHROMA_LAW_PATH       = "./chroma_law_db"
CHROMA_LAW_COLLECTION = "law_collection"
LAWFORMER_MODEL       = "thunlp/Lawformer"
MAX_SEQ_LEN           = 4096
TOP_K_LAWS            = 5
TOP_K_CHUNKS          = 30

# 知识图谱及模型路径
MODEL_DIR     = "./models"
KG_DIR        = "./kg_store"
EMB_DIM       = 768
TOP_K_TRIPLES = 20
MAX_CHUNKS    = 8

# 精确法条库
LAW_STRUCTURED_DB = "./law_structured.json"

# 数据与输出
DATA_DIR    = "./data"
OUT_DIR     = "model_output/zero_shot/GNN_Chain_2_Step"
LOG_FILE    = "gnn_chain_running.log"

# 重试配置（大幅降低）
MAX_RETRY          = 3     # 单次 API 调用重试次数（原15 → 3）
SAMPLE_MAX_RETRY   = 1     # 样本级重试次数（原15 → 1，即不重试）
NUM_WORKERS        = 10    # 并发线程数

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
_key_idx  = 0

def get_api_key():
    global _key_idx
    if not hasattr(_tl, "key"):
        with _key_lock:
            _tl.key = API_KEYS[_key_idx % len(API_KEYS)]
            _key_idx += 1
    return _tl.key

# ════════════════════════════════════════════════════════════
# 法律名称规范化（匹配 models/kg_store 文件名）
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
# Lawformer 检索器（法律文本语义检索）
# ════════════════════════════════════════════════════════════
class LawRetriever:
    def __init__(self):
        print(f"[LawRetriever] Loading {LAWFORMER_MODEL} on {DEVICE} ...")
        self.device = str(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(LAWFORMER_MODEL)
        self.model = AutoModel.from_pretrained(LAWFORMER_MODEL).to(self.device)
        self.model.eval()
        self._lock = threading.Lock()
        db_path = Path(CHROMA_LAW_PATH)
        if not db_path.exists():
            raise FileNotFoundError(f"Lawformer ChromaDB not found: {db_path}")
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_collection(CHROMA_LAW_COLLECTION)
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
            name = meta.get("law_name") or meta.get("doc", "unknown")
            if name not in best or sim > best[name]["similarity"]:
                best[name] = {"law_name": name, "similarity": sim, "matched_text": doc}
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
# KG 数据结构与 CompGCN 模型（与生成脚本一致）
# ════════════════════════════════════════════════════════════
class KG:
    def __init__(self):
        self.name = ""; self.triplets = []
        self.entity2id = {}; self.id2entity = {}
        self.rel2id = {};    self.id2rel = {}

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.name      = d["name"]
        self.triplets  = d["triplets"]
        self.entity2id = d["entity2id"]
        self.rel2id    = d["rel2id"]
        self.id2entity = {int(v): k for k, v in self.entity2id.items()}
        self.id2rel    = {int(v): k for k, v in self.rel2id.items()}

    def indexed(self):
        return [(self.entity2id[h], self.rel2id[r], self.entity2id[t])
                for h, r, t in self.triplets
                if h in self.entity2id and r in self.rel2id and t in self.entity2id]

class CompGCNLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_self = torch.nn.Linear(dim, dim)
        self.W_nei  = torch.nn.Linear(dim, dim)
        self.W_rel  = torch.nn.Linear(dim, dim)
        self.act    = torch.nn.ReLU()

    def forward(self, ent, rel, edge_index, edge_type):
        src, dst = edge_index
        msg = ent[src] * torch.sigmoid(rel[edge_type])
        agg = torch.zeros_like(ent)
        agg.index_add_(0, dst, msg)
        return self.act(self.W_self(ent) + self.W_nei(agg)), self.W_rel(rel)

class CompGCN(torch.nn.Module):
    def __init__(self, num_ent: int, num_rel: int, init_emb: torch.Tensor):
        super().__init__()
        self.ent    = torch.nn.Embedding.from_pretrained(init_emb, freeze=False)
        self.rel    = torch.nn.Embedding(num_rel, EMB_DIM)
        self.layers = torch.nn.ModuleList([CompGCNLayer(EMB_DIM), CompGCNLayer(EMB_DIM)])

    def encode(self, edge_index, edge_type):
        x, r = self.ent.weight, self.rel.weight
        for layer in self.layers:
            x, r = layer(x, r, edge_index, edge_type)
        return x, r

    def score(self, h, r, t):
        return (h * r * t).sum(-1)

# ════════════════════════════════════════════════════════════
# KG 加载与三元组评分（带缓存）
# ════════════════════════════════════════════════════════════
_kg_model_cache: dict = {}
_kg_model_cache_lock = threading.Lock()

def load_model_and_kg(law_name: str):
    mp = os.path.join(MODEL_DIR, f"{law_name}_compgcn.pt")
    kp = os.path.join(KG_DIR,    f"{law_name}_kg.json")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"模型不存在: {mp}")
    if not os.path.exists(kp):
        raise FileNotFoundError(f"KG不存在:  {kp}")

    kg = KG()
    kg.load(kp)

    init_emb = torch.zeros((len(kg.entity2id), EMB_DIM), dtype=torch.float32)
    model = CompGCN(len(kg.entity2id), len(kg.rel2id), init_emb).to(DEVICE)
    ckpt  = torch.load(mp, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, kg

def _build_edge_tensors(kg: KG):
    src, dst, rel = [], [], []
    for h, r, t in kg.indexed():
        src += [h, t]; dst += [t, h]; rel += [r, r]
    return (torch.tensor([src, dst], device=DEVICE),
            torch.tensor(rel, device=DEVICE))

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
# 多跳推理（实体种子匹配）
# ════════════════════════════════════════════════════════════
def _extract_entity_seeds(case_text: str, kg: KG, top_n: int = 10) -> list:
    seeds = []
    for entity in kg.entity2id:
        if len(entity) >= 2 and entity in case_text:
            seeds.append(entity)
        if len(seeds) >= top_n:
            break
    if not seeds:
        top = get_triples_scored(kg.name)
        seeds = list({t["head"] for t in top[:5]})
    return seeds

def get_subgraph_edges(kg: KG, seeds: list, hops: int = 2) -> list:
    nodes = set(seeds)
    edges = []
    for _ in range(hops):
        new_nodes = set()
        for h, r, t in kg.triplets:
            if h in nodes or t in nodes:
                edges.append((h, r, t))
                new_nodes.add(h); new_nodes.add(t)
        nodes |= new_nodes
    return edges

def find_reasoning_paths(edges: list, seeds: list,
                          model: CompGCN, kg: KG,
                          ei: torch.Tensor, et: torch.Tensor,
                          depth: int = 3, top_n: int = 5) -> list:
    with torch.no_grad():
        ent, rlt = model.encode(ei, et)

    adj: dict = {}
    for h, r, t in edges:
        adj.setdefault(h, []).append((r, t))

    scored = []

    def dfs(cur, path_nodes, path_rels, score, d):
        if d > 0:
            scored.append({"nodes": path_nodes[:], "rels": path_rels[:],
                           "score": round(score, 4)})
        if d >= depth:
            return
        for r_s, t_s in adj.get(cur, []):
            h_id = kg.entity2id.get(cur)
            r_id = kg.rel2id.get(r_s)
            t_id = kg.entity2id.get(t_s)
            if None in (h_id, r_id, t_id):
                continue
            step = model.score(ent[h_id].unsqueeze(0),
                               rlt[r_id].unsqueeze(0),
                               ent[t_id].unsqueeze(0)).item()
            dfs(t_s, path_nodes + [t_s], path_rels + [r_s], score + step, d + 1)

    for seed in seeds[:5]:
        dfs(seed, [seed], [], 0.0, 0)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

# ════════════════════════════════════════════════════════════
# 精确法条库（law_structured.json）
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
                    key   = f"{rec_law}_{rec_art}"
                    if chunk and key not in seen:
                        results.append(chunk); seen.add(key)
                    break
            if len(results) >= MAX_CHUNKS:
                break
        if len(results) >= MAX_CHUNKS:
            break
    return results

# ════════════════════════════════════════════════════════════
# DeepSeek API 调用（单次，带重试）
# ════════════════════════════════════════════════════════════
def call_deepseek(prompt: str, max_retry: int = MAX_RETRY) -> str:
    for attempt in range(max_retry):
        try:
            conn = http.client.HTTPSConnection(API_HOST)
            payload = json.dumps({
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
            })
            headers = {"Authorization": f"Bearer {get_api_key()}",
                       "Content-Type":  "application/json"}
            conn.request("POST", "/v1/chat/completions", payload, headers)
            raw  = conn.getresponse().read().decode("utf-8")
            data = json.loads(raw)
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            else:
                logging.error(f"API error: {data}")
        except Exception as e:
            logging.error(f"API call attempt {attempt+1} failed: {e}")
            if attempt == max_retry - 1:
                raise
            time.sleep(2)
    raise RuntimeError("API call failed after retries")

# ════════════════════════════════════════════════════════════
# Stage-1 Prompt（结构化 JSON）
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

def _fmt_laws(laws: list) -> str:
    return "\n".join(
        f"{i+1}. 【{l['law_name']}】相似度 {l['similarity']}\n   {l['matched_text'][:200]}"
        for i, l in enumerate(laws)
    ) or "  （无检索结果）"

def _fmt_triples(triples: list) -> str:
    if not triples:
        return ""
    return "\n".join(
        f"  {t['head']} --[{t['relation']}]--> {t['tail']}  (score:{t['score']})"
        for t in triples[:20]
    )

def _fmt_paths(paths: list) -> str:
    if not paths:
        return ""
    lines = []
    for p in paths:
        nodes, rels = p["nodes"], p["rels"]
        chain = "".join(f"{nodes[i]} --[{r}]--> " for i, r in enumerate(rels))
        chain += nodes[-1] if nodes else ""
        lines.append(f"  {chain}  (score={p['score']})")
    return "\n".join(lines)

def build_stage1_prompt(laws_ctx, triples_ctx, paths_ctx, full_text):
    triples_section = f"\n═══ CompGCN 知识图谱三元组（Top评分）═══\n{triples_ctx}" if triples_ctx else ""
    paths_section = f"\n═══ 多跳推理路径 ═══\n{paths_ctx}" if paths_ctx else ""
    return f"""{STAGE1_SYSTEM_BLOCK}

═══ Lawformer 检索到的相关法律 ═══
{laws_ctx}
{triples_section}
{paths_section}
═══ 题目内容 ═══
{full_text}"""

def parse_stage1_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析 Stage-1 JSON: {raw[:300]}")

# ════════════════════════════════════════════════════════════
# Stage-2 Prompt（task-aware，强调题干为核心，注入 Stage-1 推理链）
# ════════════════════════════════════════════════════════════
def _fmt_precise(chunks: list) -> str:
    return ("\n\n---\n\n".join(chunks)
            if chunks else "  （未检索到精确法条，请基于通用法律知识作答）")

def build_choice_prompt(full_text, precise, laws, triples, stage1_reasoning):
    kg_section = f"\n═══ 知识图谱三元组（仅供参考）═══\n{_fmt_triples(triples)}" if triples else ""
    return f"""你是专业法律专家。请根据以下精确法条原文和知识图谱信息，回答选择题。

【重要提示】本题可能是单选题，也可能是多选题，请仔细判断每个选项，不要遗漏正确答案。
【注意】下面的知识图谱信息仅作参考，**题目内容（题干）才是你判断的核心依据**。

═══ Stage-1 推理链（参考）═══
{stage1_reasoning}

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

═══ Lawformer 相关法律（辅助参考）═══
{_fmt_laws(laws)}
{kg_section}

═══ 题目（核心） ═══
{full_text}

分析步骤：
1. 仔细阅读题干，理解法律问题
2. 对照精确法条判断每个选项正误
3. 确认是否有多个正确选项

只输出正确选项的字母，多选用连续字母（如 A、BC、ABCD），不要解释。"""

def build_ner_prompt(full_text, precise, laws, triples, stage1_reasoning):
    kg_section = f"\n═══ 知识图谱三元组（仅供参考）═══\n{_fmt_triples(triples)}" if triples else ""
    return f"""你是法律信息抽取专家。请从文本中抽取命名实体。

【注意】下面的知识图谱信息仅作参考，**题目内容（题干）才是你抽取的核心依据**。

═══ Stage-1 推理链（参考）═══
{stage1_reasoning}

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

{kg_section}

═══ 题目（核心） ═══
{full_text}

要求：严格按照题目给定的实体类型列表分类，只抽取文中实际出现的实体，只输出选项字母。"""

def build_relation_prompt(full_text, precise, laws, triples, stage1_reasoning):
    kg_section = f"\n═══ 知识图谱三元组（仅供参考）═══\n{_fmt_triples(triples)}" if triples else ""
    return f"""你是法律关系抽取专家。请从文本中抽取关系三元组。

【注意】下面的知识图谱信息仅作参考，**题目内容（题干）才是你抽取的核心依据**。

═══ Stage-1 推理链（参考）═══
{stage1_reasoning}

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

{kg_section}

═══ 题目（核心） ═══
{full_text}

要求：关系类型严格限于题目给定的列表，只输出选项字母。"""

def build_summary_prompt(full_text, precise, laws, stage1_reasoning):
    return f"""你是法律文书编辑。请生成不超过400字的简明摘要。

【注意】下面的法律信息仅作参考，**题目内容（题干）才是你摘要的核心依据**。

═══ Stage-1 推理链（参考）═══
{stage1_reasoning}

═══ 相关法条 ═══
{_fmt_precise(precise)}

═══ 参考法律 ═══
{_fmt_laws(laws)}

═══ 案情原文（核心） ═══
{full_text}

要求：保留核心当事人、时间、争议焦点和裁判结果，不得添加原文没有的信息。

摘要:"""

def build_judgment_prompt(full_text, precise, laws, triples, suffix, stage1_reasoning):
    kg_section = f"\n═══ 知识图谱三元组（仅供参考）═══\n{_fmt_triples(triples)}" if triples else ""
    return f"""你是资深法律专家。请根据精确法条和案情事实给出准确判断。

【注意】下面的知识图谱信息仅作参考，**题目内容（题干）才是你判断的核心依据**。

═══ Stage-1 推理链（参考）═══
{stage1_reasoning}

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

═══ Lawformer 相关法律 ═══
{_fmt_laws(laws)}
{kg_section}

═══ 案情及问题（核心） ═══
{full_text}

量刑参考：犯罪数额与量刑幅度、从轻/从重情节、是否初犯退赃、检察院量刑建议。
对于选择题：只输出选项字母。对于问答题：语言专业准确，引用具体法条编号。

{suffix}"""

def build_defense_prompt(full_text, precise, laws, stage1_reasoning):
    return f"""你是法律辩护专家。请阅读诉方观点，从选项中找出与之直接对应的辩方论点。

【注意】下面的法律信息仅作参考，**题目内容（题干）才是你匹配的核心依据**。

═══ Stage-1 推理链（参考）═══
{stage1_reasoning}

═══ 精确法条原文 ═══
{_fmt_precise(precise)}

═══ 参考法律 ═══
{_fmt_laws(laws)}

═══ 题目（核心） ═══
{full_text}

匹配标准：辩方应针对诉方具体主张回应，优先选择承认部分事实但争辩法律责任的选项。
只输出选项字母。"""

def build_translation_prompt(full_text, stage1_reasoning):
    return f"""你是精通中英文法律术语的专业译者。

将给定法律术语准确翻译为中文，使用中国法律实践中的通用译法，只输出翻译结果。

═══ Stage-1 推理链（参考）═══
{stage1_reasoning}

{full_text}

翻译结果:"""

SUFFIX_MAP = {"5_2": "裁判分析过程:"}

def build_stage2_prompt(task_type, full_text, precise_chunks, laws, triples, task_name, stage1_reasoning):
    if task_type == "choice":
        return build_choice_prompt(full_text, precise_chunks, laws, triples, stage1_reasoning)
    elif task_type == "ner":
        return build_ner_prompt(full_text, precise_chunks, laws, triples, stage1_reasoning)
    elif task_type == "relation":
        return build_relation_prompt(full_text, precise_chunks, laws, triples, stage1_reasoning)
    elif task_type == "summary":
        return build_summary_prompt(full_text, precise_chunks, laws, stage1_reasoning)
    elif task_type == "judgment":
        return build_judgment_prompt(full_text, precise_chunks, laws, triples,
                                     SUFFIX_MAP.get(task_name, "答案:"), stage1_reasoning)
    elif task_type == "defense":
        return build_defense_prompt(full_text, precise_chunks, laws, stage1_reasoning)
    elif task_type == "translation":
        return build_translation_prompt(full_text, stage1_reasoning)
    else:
        return build_choice_prompt(full_text, precise_chunks, laws, triples, stage1_reasoning)

# ════════════════════════════════════════════════════════════
# task_type 猜测（Stage-1 失败降级用）
# ════════════════════════════════════════════════════════════
_MCQ_RE = re.compile(
    r"选择题|单选|多选|下列.*?(?:正确|错误|不正确|不属于|不符合)|"
    r"(?:^|\n)[A-D][：:．.]\s",
    re.MULTILINE
)

def _guess_task_type(instruction: str, input_text: str) -> str:
    combined = instruction + input_text
    if _MCQ_RE.search(combined):                          return "choice"
    if "实体" in combined or "命名" in combined:          return "ner"
    if "三元组" in combined or "关系" in combined:        return "relation"
    if "摘要" in combined:                                return "summary"
    if "翻译" in combined:                                return "translation"
    if "辩方" in combined or "辩护" in combined:         return "defense"
    return "judgment"

# ════════════════════════════════════════════════════════════
# 单条处理核心逻辑（不含样本级重试）
# ════════════════════════════════════════════════════════════
def process_item(item: dict, task_name: str) -> dict:
    instruction = item.get("instruction", "")
    input_text  = item.get("input", "")
    answer      = item.get("answer", "")
    full_text   = (instruction + "\n" + input_text).strip()

    # Step 1: Lawformer 检索
    laws: list = []
    try:
        laws = get_law_retriever().search(full_text)
    except Exception as e:
        logging.warning(f"[{task_name}] Lawformer failed: {e}")

    # Step 2: 提取法律名称并规范化
    kg_names_raw = [law["law_name"] for law in laws]
    kg_names = []
    for name in kg_names_raw:
        norm = normalize_law_name(name)
        if norm and norm not in kg_names:
            kg_names.append(norm)

    # Step 3: 加载 KG 并计算三元组评分 + 多跳路径
    all_triples: list = []
    all_paths:   list = []
    for law_name in kg_names:
        try:
            ts = get_triples_scored(law_name)
            for t in ts:
                t["_law"] = law_name
            all_triples.extend(ts)

            model, kg = load_model_and_kg(law_name)
            seeds = _extract_entity_seeds(full_text, kg)
            if seeds:
                sub_edges = get_subgraph_edges(kg, seeds, hops=2)
                ei, et    = _build_edge_tensors(kg)
                paths     = find_reasoning_paths(sub_edges, seeds, model, kg, ei, et)
                for p in paths:
                    p["_law"] = law_name
                all_paths.extend(paths)
        except Exception as e:
            logging.warning(f"[{task_name}] KG processing {law_name}: {e}")

    # 去重排序
    all_triples.sort(key=lambda x: x["score"], reverse=True)
    seen_t, deduped = set(), []
    for t in all_triples:
        key = (t["head"], t["relation"], t["tail"])
        if key not in seen_t:
            seen_t.add(key); deduped.append(t)
        if len(deduped) >= 40:
            break
    all_paths.sort(key=lambda x: x["score"], reverse=True)
    top_paths = all_paths[:10]

    # Step 4: Stage-1 API 调用
    p1 = build_stage1_prompt(
        _fmt_laws(laws), _fmt_triples(deduped), _fmt_paths(top_paths), full_text
    )
    stage1_raw = ""
    stage1_data = {}
    for attempt in range(MAX_RETRY):
        try:
            stage1_raw = call_deepseek(p1)
            stage1_data = parse_stage1_json(stage1_raw)
            break
        except Exception as e:
            logging.warning(f"[{task_name}] Stage-1 attempt {attempt+1}: {e}")
            if attempt == MAX_RETRY - 1:
                stage1_data = {
                    "reasoning": "Stage-1 failed",
                    "statutes":  [{"law": l["law_name"], "articles": []} for l in laws[:3]],
                    "task_type": _guess_task_type(instruction, input_text),
                }
            else:
                time.sleep(1)

    # Step 5: 精确法条检索
    statutes    = stage1_data.get("statutes", [])
    task_type   = stage1_data.get("task_type", "choice")
    precise_cks = retrieve_precise_chunks(statutes)
    stage1_reasoning = stage1_data.get("reasoning", "（无推理过程）")

    logging.info(
        f"[{task_name}] task_type={task_type} | kg_names={kg_names} | "
        f"triples={len(deduped)} | paths={len(top_paths)} | precise={len(precise_cks)}"
    )

    # Step 6: Stage-2 API 调用（强调题干核心，注入 Stage-1 推理链，KG 为空时不显示）
    p2 = build_stage2_prompt(task_type, full_text, precise_cks, laws, deduped, task_name, stage1_reasoning)
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
        "stage1_reasoning": stage1_reasoning,
        "stage1_statutes":  statutes,
        "precise_chunks":   precise_cks,
        "retrieved_laws":   [{"law_name": l["law_name"], "similarity": l["similarity"],
                               "matched_text": l["matched_text"][:200]} for l in laws],
        "kg_names":   kg_names,
        "kg_triples": [{"head": t["head"], "relation": t["relation"],
                        "tail": t["tail"], "score": t["score"],
                        "law":  t.get("_law", "")} for t in deduped[:20]],
        "kg_paths":   top_paths[:5],
    }

# ════════════════════════════════════════════════════════════
# 样本级自动重试包装器（已关闭重试）
# ════════════════════════════════════════════════════════════
def process_item_with_retry(item: dict, task_name: str) -> dict:
    try:
        result = process_item(item, task_name)
        return result
    except Exception as e:
        logging.error(f"[{task_name}] 样本处理异常: {e}")
        return {
            "input": item.get("input", ""),
            "output": "FAILED",
            "answer": item.get("answer", ""),
            "error": str(e)
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
        raw = json.load(open(path, encoding="utf-8"))
        if raw and isinstance(raw[0], dict) and "first_item" in raw[0]:
            data = [item["first_item"] for item in raw if "first_item" in item]
        else:
            data = raw
    return data

def run_on_file(f_path: str):
    task_name = os.path.basename(f_path).split(".")[0]
    data = load_data(f_path)
    print(f"\n🚀  {task_name} | {len(data)} items")
    logging.info(f"Start {task_name}: {len(data)} items")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(process_item_with_retry, item, task_name): item for item in data}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=task_name):
            try:
                results.append(fut.result())
            except Exception as e:
                logging.error(f"[{task_name}] 未捕获异常: {e}")
                results.append({
                    "input": futs[fut].get("input", ""),
                    "output": "FAILED",
                    "answer": futs[fut].get("answer", ""),
                    "error": str(e)
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"gnn_chain_{task_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    failed = sum(1 for r in results if r.get("output") == "FAILED")
    empty_kg = sum(1 for r in results if not r.get("kg_triples"))
    no_precise = sum(1 for r in results if not r.get("precise_chunks"))
    print(f"✅  Saved → {out_path}\n"
          f"    failed={failed}/{len(results)}  "
          f"kg_triples空={empty_kg}/{len(results)}  "
          f"precise空={no_precise}/{len(results)}")

def main():
    print("=" * 65)
    print("GNN-Chain Pipeline (速度优化版)")
    print("  - 知识图谱匹配：Lawformer → 直接加载 CompGCN")
    print("  - 样本级自动重试 → 关闭（FAILED 直接返回）")
    print("  - Stage-2 注入 Stage-1 推理链，KG 为空时不显示")
    print("  - API 内部重试次数降低为 3 次")
    print("=" * 65)

    load_law_structured()
    print("\nInitializing Lawformer ...")
    get_law_retriever()

    avail = list_available_kg_names()
    print(f"\n可用 KG 文件: {len(avail)} 个")
    if avail:
        print(f"  示例: {avail[:5]}")
    else:
        print("  ⚠️  未找到任何 KG 文件，请检查 MODEL_DIR 和 KG_DIR 路径配置")

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