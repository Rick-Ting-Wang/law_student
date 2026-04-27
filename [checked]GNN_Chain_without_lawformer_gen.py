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

API_HOST  = "dpapi.cn"
MODEL_NAME = "deepseek-v3"

# ── 法律检索路径（text2vec，替换原 Lawformer）────────────────
CHROMA_LAW_PATH       = "./chroma_db"          # 新向量库路径
CHROMA_LAW_COLLECTION = "china_law"            # 新集合名称
TEXT2VEC_MODEL        = "shibing624/text2vec-base-chinese"
TOP_K_LAWS   = 1   # 只取相似度最高的 1 条法律
TOP_K_CHUNKS = 30  # 先召回 30 个 chunk，再聚合

# ── 知识图谱及模型路径 ──────────────────────────────────────
MODEL_DIR    = "./models"
KG_DIR       = "./kg_store"
EMB_DIM      = 768
TOP_K_TRIPLES = 3   # 只取评分最高的 3 个三元组
MAX_CHUNKS    = 8

# ── 精确法条库 ──────────────────────────────────────────────
LAW_STRUCTURED_DB = "./law_structured.json"

# ── 数据与输出 ──────────────────────────────────────────────
DATA_DIR = "./data"
OUT_DIR  = "./model_output/zero_shot/GNN_Chain_without_lawformer"
LOG_FILE = "gnn_chain_running.log"

# ── 重试 / 并发配置 ─────────────────────────────────────────
MAX_RETRY   = 3
NUM_WORKERS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════
# 日志与 API Key 管理
# ════════════════════════════════════════════════════════════
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_tl       = threading.local()
_key_lock = threading.Lock()
_key_idx  = 0


def get_api_key():
    global _key_idx
    if not hasattr(_tl, "key"):
        with _key_lock:
            _tl.key  = API_KEYS[_key_idx % len(API_KEYS)]
            _key_idx += 1
    return _tl.key


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
    """
    用 shibing624/text2vec-base-chinese 编码查询，
    在 ChromaDB (china_law 集合) 中检索最相关的法律条款。

    返回格式（与原 Lawformer 版对齐）：
    [
      {
        "law_name":    str,   # 法律名称
        "similarity":  float, # 余弦相似度 [0, 1]
        "matched_text": str,  # 最佳匹配 chunk
        "article":     str,   # 条款号
        "chapter":     str,   # 章节
        "publish_date": str,  # 发布日期
      }, ...
    ]
    """

    def __init__(self):
        print(f"[LawRetriever] Loading {TEXT2VEC_MODEL} ...")
        # SentenceTransformer 本身管理设备，无需手动 .to(device)
        self.model = SentenceTransformer(TEXT2VEC_MODEL)

        db_path = Path(CHROMA_LAW_PATH)
        if not db_path.exists():
            raise FileNotFoundError(f"ChromaDB not found: {db_path}")
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_collection(CHROMA_LAW_COLLECTION)

        print(f"[LawRetriever] Ready — {self.collection.count()} chunks loaded")

    # ── 编码（线程安全：SentenceTransformer.encode 本身是无状态的）──
    def _encode(self, text: str) -> list:
        return self.model.encode(text, convert_to_tensor=False).tolist()

    # ── 按法律名称聚合，保留每部法律相似度最高的 chunk ────────
    def _aggregate(self, res: dict) -> list:
        best: dict = {}
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            sim      = round(1 - dist, 4)          # 余弦相似度
            law_name = meta.get("law", "unknown")   # 字段名：law

            if law_name not in best or sim > best[law_name]["similarity"]:
                best[law_name] = {
                    "law_name":     law_name,
                    "similarity":   sim,
                    "matched_text": doc,                         # chunk 文本
                    "article":      meta.get("article", ""),
                    "chapter":      meta.get("chapter", ""),
                    "publish_date": meta.get("publish_date", ""),
                }

        return sorted(
            best.values(), key=lambda x: x["similarity"], reverse=True
        )[:TOP_K_LAWS]

    # ── 对外接口 ─────────────────────────────────────────────
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


# ── 单例（线程安全懒加载）────────────────────────────────────
_law_retriever      = None
_law_retriever_lock = threading.Lock()


def get_law_retriever() -> LawRetriever:
    global _law_retriever
    if _law_retriever is None:
        with _law_retriever_lock:
            if _law_retriever is None:
                _law_retriever = LawRetriever()
    return _law_retriever


# ════════════════════════════════════════════════════════════
# KG 数据结构与 CompGCN 模型（保持不变）
# ════════════════════════════════════════════════════════════
class KG:
    def __init__(self):
        self.name      = ""
        self.triplets  = []
        self.entity2id = {}; self.id2entity = {}
        self.rel2id    = {}; self.id2rel    = {}

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
        return [
            (self.entity2id[h], self.rel2id[r], self.entity2id[t])
            for h, r, t in self.triplets
            if h in self.entity2id and r in self.rel2id and t in self.entity2id
        ]


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
# KG 加载与三元组评分（保持不变）
# ════════════════════════════════════════════════════════════
_kg_model_cache      : dict = {}
_kg_model_cache_lock = threading.Lock()


def load_model_and_kg(law_name: str):
    mp = os.path.join(MODEL_DIR, f"{law_name}_compgcn.pt")
    kp = os.path.join(KG_DIR,   f"{law_name}_kg.json")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"模型不存在: {mp}")
    if not os.path.exists(kp):
        raise FileNotFoundError(f"KG不存在:  {kp}")

    kg = KG()
    kg.load(kp)

    init_emb = torch.zeros((len(kg.entity2id), EMB_DIM), dtype=torch.float32)
    model    = CompGCN(len(kg.entity2id), len(kg.rel2id), init_emb).to(DEVICE)
    ckpt     = torch.load(mp, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, kg


def _build_edge_tensors(kg: KG):
    src, dst, rel = [], [], []
    for h, r, t in kg.indexed():
        src += [h, t]; dst += [t, h]; rel += [r, r]
    return (
        torch.tensor([src, dst], device=DEVICE),
        torch.tensor(rel,        device=DEVICE),
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
    rows   = []
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
# 精确法条库（保持不变）
# ════════════════════════════════════════════════════════════
_law_structured      : list = []
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
# DeepSeek API 调用（保持不变）
# ════════════════════════════════════════════════════════════
def call_deepseek(prompt: str, max_retry: int = MAX_RETRY) -> str:
    for attempt in range(max_retry):
        try:
            conn    = http.client.HTTPSConnection(API_HOST)
            payload = json.dumps({
                "model":    MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
            })
            headers = {
                "Authorization": f"Bearer {get_api_key()}",
                "Content-Type":  "application/json",
            }
            conn.request("POST", "/v1/chat/completions", payload, headers)
            raw  = conn.getresponse().read().decode("utf-8")
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
# Prompt 构建（选择题专用，单次 API 调用）
# ════════════════════════════════════════════════════════════
def build_choice_prompt(
    full_text:   str,
    precise_cks: list,
    laws:        list,
    triples:     list,
) -> str:
    """
    参数：
      full_text   — 题目完整文本
      precise_cks — 精确法条列表
      laws        — text2vec 检索结果（最多 1 个），字段含 law_name / matched_text / article 等
      triples     — CompGCN 三元组（最多 3 个）
    """

    # ── 精确法条 ──────────────────────────────────────────
    precise_section = ""
    if precise_cks:
        precise_section = "═══ 精确法条原文 ═══\n"
        for i, chunk in enumerate(precise_cks, 1):
            precise_section += f"\n【法条 {i}】\n{chunk}\n"
        precise_section += "\n"

    # ── 向量检索法律（无精确命中时显示）──────────────────
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

    # ── 知识图谱三元组（无精确命中时显示）────────────────
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


# ════════════════════════════════════════════════════════════
# 单条处理核心逻辑（单次 API 调用）
# ════════════════════════════════════════════════════════════
def process_item(item: dict, task_name: str) -> dict:
    instruction = item.get("instruction", "")
    input_text  = item.get("input", "")
    answer      = item.get("answer", "")
    full_text   = (instruction + "\n" + input_text).strip()

    # Step 1: text2vec 检索 → 只取相似度最高的 1 条
    laws: list = []
    try:
        laws = get_law_retriever().search(full_text)   # 已在 _aggregate 中限制 TOP_K_LAWS=1
    except Exception as e:
        logging.warning(f"[{task_name}] LawRetriever search failed: {e}")

    # Step 2: 提取法律名称并规范化（对应 KG 文件名）
    kg_names: list = []
    for law in laws:
        norm = normalize_law_name(law["law_name"])
        if norm and norm not in kg_names:
            kg_names.append(norm)

    # Step 3: CompGCN 三元组评分 → 只取最高的 3 个
    all_triples: list = []
    for law_name in kg_names:
        try:
            all_triples.extend(get_triples_scored(law_name))
        except Exception as e:
            logging.warning(f"[{task_name}] KG processing {law_name}: {e}")

    all_triples.sort(key=lambda x: x["score"], reverse=True)
    top_triples = all_triples[:TOP_K_TRIPLES]

    # Step 4: 精确法条检索（此处预留接口，默认为空）
    precise_cks: list = []

    # Step 5: 精确命中时清空 KG（减少噪声）
    if precise_cks:
        top_triples = []
        logging.info(f"[{task_name}] 精确命中法条 {len(precise_cks)} 条，清空KG")

    logging.info(
        f"[{task_name}] laws={len(laws)} | kg_names={kg_names} | "
        f"triples={len(top_triples)} | precise={len(precise_cks)}"
    )

    # Step 6: 单次 API 调用
    prompt = build_choice_prompt(full_text, precise_cks, laws, top_triples)
    final  = ""
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
        "input":  full_text,
        "output": final,
        "answer": answer,
        # 统一用 law_name 字段输出（已在 _aggregate 中对齐）
        "retrieved_laws": [
            {"law_name": l["law_name"], "similarity": l["similarity"]}
            for l in laws
        ],
        "kg_names":  kg_names,
        "kg_triples": [
            {"head": t["head"], "relation": t["relation"],
             "tail": t["tail"], "score":    t["score"]}
            for t in top_triples
        ],
        "precise_chunks": precise_cks,
        "api_calls": 1,
    }


# ════════════════════════════════════════════════════════════
# 样本处理包装器（保持不变）
# ════════════════════════════════════════════════════════════
def process_item_with_retry(item: dict, task_name: str) -> dict:
    try:
        return process_item(item, task_name)
    except Exception as e:
        logging.error(f"[{task_name}] 处理异常: {e}")
        return {
            "input":  item.get("input", ""),
            "output": "FAILED",
            "answer": item.get("answer", ""),
            "error":  str(e),
            "api_calls": 0,
        }


# ════════════════════════════════════════════════════════════
# 文件处理与主流程（保持不变）
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


def run_on_file(f_path: str):
    task_name = os.path.basename(f_path).split(".")[0]
    data      = load_data(f_path)

    if not data:
        print(f"[SKIP] {task_name}: 无数据")
        return

    print(f"\n[RUN]  {task_name} | {len(data)} items")
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
                    "input":  futs[fut].get("input", ""),
                    "output": "FAILED",
                    "answer": futs[fut].get("answer", ""),
                    "error":  str(e),
                    "api_calls": 0,
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"gnn_chain_{task_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    failed          = sum(1 for r in results if r.get("output") == "FAILED")
    empty_kg        = sum(1 for r in results if not r.get("kg_triples"))
    no_precise      = sum(1 for r in results if not r.get("precise_chunks"))
    total_api_calls = sum(r.get("api_calls", 0) for r in results)

    print(f"[DONE] Saved -> {out_path}")
    print(f"       failed:        {failed}/{len(results)}")
    print(f"       kg_empty:      {empty_kg}/{len(results)}")
    print(f"       precise_empty: {no_precise}/{len(results)}")
    print(f"       total_api_calls: {total_api_calls} "
          f"(avg {total_api_calls / len(results):.1f}/sample)")


def main():
    print("=" * 70)
    print("GNN-Chain Pipeline - text2vec-base-chinese 版")
    print("=" * 70)
    print("  1. 替换 Lawformer -> shibing624/text2vec-base-chinese")
    print("  2. ChromaDB: ./chroma_db / china_law")
    print("  3. 单次 API 调用")
    print("  4. 只取相似度最高的 1 条法律 + top-3 KG 三元组")
    print("=" * 70)

    load_law_structured()

    print("\n[INIT] Loading LawRetriever ...")
    get_law_retriever()

    avail = list_available_kg_names()
    print(f"\n[KG]  可用 KG 文件: {len(avail)} 个")
    if avail:
        print(f"      示例: {', '.join(avail[:3])}")
    else:
        print("      未找到 KG 文件，请检查 MODEL_DIR / KG_DIR 配置")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not files:
        print(f"\n[ERR] 未在 {DATA_DIR} 找到数据文件")
        return

    print(f"\n[DATA] 找到 {len(files)} 个任务文件\n")
    for f in files:
        run_on_file(f)

    print("\n" + "=" * 70)
    print("Pipeline 处理完成！")
    print("=" * 70)
    logging.info("GNN-Chain complete")


if __name__ == "__main__":
    main()