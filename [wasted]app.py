"""
[wasted]app.py
======
依赖 (仅需两个包):
    pip install fastapi uvicorn

目录结构:
    project/
    ├── [wasted]app.py
    └── templates/
        └── index.html

启动:
    python [wasted]app.py
    或
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

访问:  http://127.0.0.1:8000
"""

import json
import logging
import http.client
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "templates" / "index.html"

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="Legal Intelligence Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve HTML ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    if not HTML_FILE.exists():
        return HTMLResponse(
            content=(
                f"<h2>Error: templates/index.html not found</h2>"
                f"<p>Expected path: <code>{HTML_FILE}</code></p>"
                f"<p>[wasted]app.py location: <code>{BASE_DIR}</code></p>"
            ),
            status_code=500,
        )
    return HTMLResponse(content=HTML_FILE.read_text(encoding="utf-8"))

# ── Health ───────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status":     "ok",
        "base_dir":   str(BASE_DIR),
        "html_found": HTML_FILE.exists(),
    }

# ── Schemas ──────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str

class TriplesRequest(BaseModel):
    law_name: str
    top_n: Optional[int] = 50

class ReasoningRequest(BaseModel):
    case_text: str
    laws:      List[dict]
    triples:   List[dict]

class AnalysisRequest(BaseModel):
    case_text:      str
    laws:           List[dict]
    reasoning_path: str

# ── DeepSeek ─────────────────────────────────────────────────────
DEEPSEEK_HOST  = "dpapi.cn"
DEEPSEEK_MODEL = "deepseek-v3"
DEEPSEEK_KEY   = "sk-aZ4KJUKxqQGI0kvJBf87Aa2eE53a487e81A8D26f89CfD486"

def call_deepseek(messages: list) -> str:
    conn = http.client.HTTPSConnection(DEEPSEEK_HOST, timeout=120)
    payload = json.dumps({"model": DEEPSEEK_MODEL, "messages": messages})
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type":  "application/json",
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    raw  = conn.getresponse().read().decode("utf-8")
    data = json.loads(raw)
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data["choices"][0]["message"]["content"]


def stream_deepseek(messages: list):
    try:
        conn = http.client.HTTPSConnection(DEEPSEEK_HOST, timeout=120)
        payload = json.dumps({
            "model":    DEEPSEEK_MODEL,
            "messages": messages,
            "stream":   True,
        })
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_KEY}",
            "Content-Type":  "application/json",
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        resp   = conn.getresponse()
        buffer = ""
        while True:
            chunk = resp.read(512)
            if not chunk:
                break
            buffer += chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                chunk_str = line[5:].strip()
                if chunk_str == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                try:
                    delta = (
                        json.loads(chunk_str)["choices"][0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield f"data: {json.dumps({'content': delta})}\n\n"
                except Exception:
                    continue
    except Exception as e:
        logger.exception("stream_deepseek error")
        yield f"data: {json.dumps({'content': f'[API Error: {e}]'})}\n\n"
        yield "data: [DONE]\n\n"

# ── /api/search ──────────────────────────────────────────────────
@app.post("/api/search")
def search_laws(req: SearchRequest):
    logger.info(f"[search] {req.query[:80]}")

    # ── 接入真实 Lawformer 后取消注释 ──────────────
    # from law_retriever import LawRetriever
    # laws = LawRetriever().search(req.query)
    # return {"success": True, "laws": laws}

    mock = [
        {
            "law_name":     "中华人民共和国刑法",
            "filename":     "criminal_law.txt",
            "similarity":   0.9234,
            "chunk_index":  4,
            "total_chunks": 120,
            "matched_text": (
                "第二百三十三条　过失致人死亡的，处三年以上七年以下有期徒刑；"
                "情节较轻的，处三年以下有期徒刑。本法另有规定的，依照规定。"
            ),
        },
        {
            "law_name":     "中华人民共和国枪支管理法",
            "filename":     "firearms_law.txt",
            "similarity":   0.8721,
            "chunk_index":  2,
            "total_chunks": 45,
            "matched_text": (
                "第三条　民用枪支的配置或者持有，必须符合本法规定的条件，"
                "经省级人民政府公安机关批准，领取持枪证件，方可持有。"
            ),
        },
        {
            "law_name":     "中华人民共和国刑事诉讼法",
            "filename":     "criminal_procedure.txt",
            "similarity":   0.8105,
            "chunk_index":  8,
            "total_chunks": 200,
            "matched_text": "第十二条　未经人民法院依法判决，对任何人都不得确定有罪。",
        },
    ]
    return {"success": True, "laws": mock}

# ── /api/triples ─────────────────────────────────────────────────
@app.post("/api/triples")
def get_triples(req: TriplesRequest):
    logger.info(f"[triples] {req.law_name}")

    # ── 接入真实 CompGCN 后取消注释 ────────────────
    # from kg_model import get_triples_with_scores
    # triples = get_triples_with_scores(req.law_name, req.top_n)
    # return {"success": True, "law_name": req.law_name, "triples": triples}

    mock = [
        {"head": "过失致人死亡罪", "relation": "属于条款",   "tail": "刑法第233条",      "score": 4.821},
        {"head": "过失致人死亡罪", "relation": "量刑范围",   "tail": "三至七年有期徒刑",  "score": 4.612},
        {"head": "过失行为",       "relation": "导致结果",   "tail": "他人死亡",          "score": 4.503},
        {"head": "过失致人死亡罪", "relation": "主观要件",   "tail": "过失",              "score": 4.401},
        {"head": "猎枪",           "relation": "属于类别",   "tail": "民用枪支",          "score": 4.280},
        {"head": "民用枪支",       "relation": "受规范于",   "tail": "枪支管理法",        "score": 4.155},
        {"head": "非法持有猎枪",   "relation": "违反",       "tail": "枪支管理法第3条",   "score": 3.990},
        {"head": "过失致人死亡罪", "relation": "区别于",     "tail": "故意杀人罪",        "score": 3.821},
        {"head": "情节较轻",       "relation": "适用量刑",   "tail": "三年以下有期徒刑",  "score": 3.701},
        {"head": "被告人",         "relation": "实施行为",   "tail": "过失致人死亡",      "score": 3.580},
        {"head": "误认猎物",       "relation": "认定为",     "tail": "疏忽大意过失",      "score": 3.445},
        {"head": "刑事责任",       "relation": "追究对象",   "tail": "被告人金德林",      "score": 3.310},
        {"head": "疏忽大意过失",   "relation": "对比",       "tail": "过于自信过失",      "score": 3.105},
        {"head": "刑法第233条",    "relation": "隶属于",     "tail": "刑法分则第四章",    "score": 2.980},
        {"head": "刑事案件",       "relation": "适用程序",   "tail": "刑事诉讼法",        "score": 2.810},
    ]
    top = req.top_n or 50
    for t in mock:
        t["_law"] = req.law_name
    return {"success": True, "law_name": req.law_name, "triples": mock[:top]}

# ── Prompt builders ──────────────────────────────────────────────
def _reasoning_prompt(req: ReasoningRequest) -> str:
    laws_txt = "\n".join(
        f"- {l['law_name']} (similarity: {l['similarity']})" for l in req.laws
    )
    triples_txt = "\n".join(
        f"  {t['head']} --[{t['relation']}]--> {t['tail']}  (score: {t['score']})"
        for t in req.triples[:30]
    )
    return f"""你是一位专业法律分析师。请根据以下信息，生成一条完整的法律推理路径。

【案情描述】
{req.case_text}

【匹配到的相关法律】
{laws_txt}

【知识图谱关键三元组】
{triples_txt}

请严格按照以下格式输出，每个步骤独立成段：

1. 【事实认定】
梳理案件核心事实要素（时间、地点、行为人、行为、结果）。

2. 【法律关系识别】
基于知识图谱三元组，分析涉及的法律关系链条。

3. 【法条适用】
逐步推导应适用的具体法律条文，引用条款编号。

4. 【构成要件分析】
逐一检验犯罪/违法构成要件是否满足。

5. 【推理结论】
得出初步法律定性。

每步须有逻辑依据，使用专业法律语言。"""


def _analysis_prompt(req: AnalysisRequest) -> str:
    laws_txt = "\n\n".join(
        f"【{i+1}】{l['law_name']}（相似度 {l['similarity']}）\n{l['matched_text'][:400]}"
        for i, l in enumerate(req.laws)
    )
    return f"""你是一位资深法律专家，请对以下案件出具完整的法律分析意见书。

【案情描述】
{req.case_text}

【相关法律条文节选】
{laws_txt}

【推理路径】
{req.reasoning_path}

请严格按照以下格式输出：

一、【案件概述】
二、【法律分析】
三、【罪名/责任认定】
四、【量刑/处罚建议】
五、【引用法条】
六、【综合结论】

保持专业、严谨的法律文书风格。"""

# ── /api/reasoning ───────────────────────────────────────────────
@app.post("/api/reasoning")
def generate_reasoning(req: ReasoningRequest):
    try:
        result = call_deepseek([{"role": "user", "content": _reasoning_prompt(req)}])
        return {"success": True, "reasoning_path": result}
    except Exception as e:
        logger.exception("reasoning failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reasoning/stream")
def generate_reasoning_stream(req: ReasoningRequest):
    return StreamingResponse(
        stream_deepseek([{"role": "user", "content": _reasoning_prompt(req)}]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── /api/analysis ────────────────────────────────────────────────
@app.post("/api/analysis")
def generate_analysis(req: AnalysisRequest):
    try:
        result = call_deepseek([{"role": "user", "content": _analysis_prompt(req)}])
        return {"success": True, "analysis": result}
    except Exception as e:
        logger.exception("analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analysis/stream")
def generate_analysis_stream(req: AnalysisRequest):
    return StreamingResponse(
        stream_deepseek([{"role": "user", "content": _analysis_prompt(req)}]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")