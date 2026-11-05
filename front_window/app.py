# app.py
# ============================================================
# 智研 Agent —— FastAPI 后端服务
#
# 作用：把 src/ 下的 agent.py 多 Agent 工作流封装为 HTTP API，
#       供 front_window/index.html 单页前端调用。
#       不修改后端任何已有代码，仅做引用与转发。
#
# 运行：uvicorn app:app --reload --port 8000
# 接口：
#   GET  /         -> 返回前端页面 index.html
#   POST /api/chat -> 对话入口（自动意图路由 RAG / GRAPH / VISUAL）
# ============================================================
# 延迟注解求值：允许 str | None 等新式写法在低版本分析目标下通过静态检查
from __future__ import annotations

import os
import re
import sys
import json
import uuid
import base64
import importlib.util
from typing import Any
from urllib.parse import quote

# pyright 定向抑制：IDE 分析环境解释器未装依赖时误报，运行时依赖真实存在
from fastapi import FastAPI, UploadFile, File, Form  # pyright: ignore[reportMissingImports]
from fastapi.responses import (  # pyright: ignore[reportMissingImports]
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel  # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage  # pyright: ignore[reportMissingImports]

import fitz  # pyright: ignore[reportMissingImports]  # PyMuPDF：阅读器目录树 / 内部引用链接提取

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(BASE_DIR), "src")  # agent.py 等后端模块所在处

# 后端接口版本戳：前端 index.html 里有一份同样的常量，两者不一致就提示重启服务。
# 改动抓取/设置相关接口时记得同步 +1（或改日期），否则旧进程无法被识别出来。
API_VERSION = "2026-11-05-3"

# 以脚本方式直接运行（python app.py）时，先把工作目录切到本文件所在目录：
# source_tracing.py 用相对路径 ./qdrant_db 打开向量库，不切换会新建一个空库。
if __name__ == "__main__":
    os.chdir(BASE_DIR)

# 把 src/ 加入模块搜索路径，使 agent.py 内部的 `import config` 可用
sys.path.insert(0, AGENT_DIR)

import config  # noqa: E402  加载 API Key / 模型配置

config.apply()

import dedup_service  # noqa: E402  语义查重与 HITL 版本检查
import version_dedup  # noqa: E402  版本演进 Diff（difflib 逐行对比）
import arxiv_tool  # noqa: E402  外部学术 API（arXiv 检索/下载 + GitHub 复现性）
import scholar_sources  # noqa: E402  多学术来源统一检索 + 标识符解析
import bibparse  # noqa: E402  RIS/BibTeX 批量解析
import citation_quality  # noqa: E402  引用含金量评估与应付式引用过滤
import writing  # noqa: E402  学术写作辅助（框架生成 / 拓扑图 / 中译英 / 图注 / DOI 解析）


def _load_module(mod_name: str, path: str):
    """按路径动态加载 src/ 下的 agent.py。

    注意：必须先注册到 sys.modules 再 exec_module，
    否则 langgraph 用 get_type_hints 解析 AgentState 的 Annotated 注解时
    会因找不到模块全局命名空间而抛 NameError。
    """
    spec = importlib.util.spec_from_file_location(mod_name, path)
    # 防御：路径非法时 spec 或 loader 可能为 None，提前抛出可读错误
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {mod_name}: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# 加载多 Agent 工作流（编译好的 LangGraph app 与三个子功能函数）
_agent = _load_module("agent_core", os.path.join(AGENT_DIR, "agent.py"))
graph_app = _agent.app
retrieve_with_anchors = _agent.retrieve_with_anchors
build_and_analyze_citation_graph = _agent.build_and_analyze_citation_graph
execute_data_visualization = _agent.execute_data_visualization

# RAG 溯源模块（PDF 入库用）；若加载失败（缺依赖）则导入功能不可用，对话仍可跑
_rag_mod = getattr(_agent, "_rag", None)
ingest_pdf_with_anchors = getattr(_rag_mod, "ingest_pdf_with_anchors", None)

# 图谱挖掘模块（引用边构建等纯函数）
_graph_mod = getattr(_agent, "_graph", None)


def _short_label(text: str, max_len: int = 45) -> str:
    """图谱节点短标签。"""
    text = " ".join(str(text).split())
    return text[:max_len]


def _build_real_citations() -> tuple[list[dict[str, str]], list[str]]:
    """
    从库内论文的参考文献区（is_ref 块）抽取真实引用网络：
      1. 回填每篇论文的真实标题（首页首个短块，存入登记簿缓存）；
      2. 从 Qdrant 拉取各论文的参考文献块文本；
      3. 调用图谱模块解析条目、共引聚类并建边。
    返回 (edges, 库内论文标签列表)；无数据时返回 ([], [])。
    """
    if _rag_mod is None or _graph_mod is None:
        return [], []
    from qdrant_client.models import (  # pyright: ignore[reportMissingImports]
        Filter,
        FieldCondition,
        MatchValue,
    )

    papers = _load_registry()
    if not papers:
        return [], []

    # ---- 回填真实标题（登记簿缓存，避免每次重解析 PDF） ----
    changed = False
    lib_labels = {}
    for it in papers:
        if not it.get("title"):
            path = os.path.join(UPLOAD_DIR, it["filename"])
            if os.path.exists(path):
                chunks = _rag_mod.extract_chunks_from_pdf(path)
                it["title"] = _guess_title(chunks, it["paper_id"])
                changed = True
        lib_labels[it["paper_id"]] = _short_label(it.get("title") or it["paper_id"])
    if changed:
        _save_registry(papers)

    # ---- 拉取各论文参考文献块 ----
    refs_by_paper = {}
    for pid, label in lib_labels.items():
        points, _ = _rag_mod.qdrant.scroll(
            collection_name=_rag_mod.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="paper_id", match=MatchValue(value=pid)),
                    FieldCondition(key="is_ref", match=MatchValue(value=True)),
                ]
            ),
            limit=500,
            with_payload=True,
        )
        refs_by_paper[label] = " ".join(p.payload["text"] for p in points)

    edges = _graph_mod.build_citation_edges(refs_by_paper, lib_labels)
    return edges, list(lib_labels.values())


# 演示用默认引用数据：库内无论文或未抽取到引用关系时兜底
DEFAULT_CITATIONS = [
    {"source": "BERT_2018", "target": "Transformer_2017"},
    {"source": "GPT1_2018", "target": "Transformer_2017"},
    {"source": "GPT2_2019", "target": "GPT1_2018"},
    {"source": "GPT3_2020", "target": "GPT2_2019"},
    {"source": "ViT_2020", "target": "Transformer_2017"},
    {"source": "RoBERTa_2019", "target": "BERT_2018"},
    {"source": "T5_2019", "target": "Transformer_2017"},
    {"source": "LLaMA_2023", "target": "GPT3_2020"},
]

# 演示用 CSV：不存在时自动生成一份训练日志
SAMPLE_CSV = os.path.join(BASE_DIR, "data.csv")
if not os.path.exists(SAMPLE_CSV):
    with open(SAMPLE_CSV, "w", encoding="utf-8") as f:
        f.write("Epoch,Accuracy,Loss\n")
        for epoch, (acc, loss) in enumerate(
            zip(
                [55, 63, 71, 77, 82, 86, 89, 91, 92, 93],
                [2.3, 1.8, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4],
            ),
            start=1,
        ):
            f.write(f"{epoch},{acc},{loss}\n")

app = FastAPI(title="智研 Agent 前端服务")


class ChatRequest(BaseModel):
    """对话请求体：message 必填；csv_path / citations_data 可选（不填用演示数据）。"""

    message: str
    session_id: str | None = None  # 会话 ID，用于 checkpointer 多轮状态
    csv_path: str | None = None  # VISUAL 分支：CSV 路径
    citations_data: list[dict[str, str]] | None = None  # GRAPH 分支：引用关系数据
    paper_id: str | None = None  # RAG 分支：显式指定论文（不填则自动从消息中匹配）
    # 前端「已选 N 项」：显式限定检索论文范围（多篇 -> 逐篇遍历后综合对比）
    paper_ids: list[str] | None = None
    # 前端「思考强度」：light / medium / deep，深度会扩大溯源检索范围
    thinking_level: str | None = None


def _match_paper_ids(message: str) -> list[str]:
    """从登记簿中匹配用户消息提到的所有论文，返回 paper_id 列表。

    匹配规则（任一命中即收录）：
      1. 消息完整包含 paper_id、文件名（去扩展名）或论文标题
         （右击知识库装入的《标题》引用即走此规则）；
      2. 消息包含 paper_id 的前 12 个字符——覆盖前端把长文件名
         截断显示后用户复制提问的场景（如 "1cvpr-2026-T-MOR-Lea"）。
    比较前统一小写并去掉空格/下划线，提升容错。
    """

    def norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "")

    msg = norm(message)
    hits = []
    for it in _load_registry():
        pid = it["paper_id"]
        stem = os.path.splitext(it["filename"])[0]
        title = it.get("title") or ""
        if norm(pid) in msg or norm(stem) in msg or (title and norm(title) in msg):
            hits.append(pid)
        elif len(pid) >= 12 and norm(pid[:12]) in msg:
            hits.append(pid)
    return hits


def _match_paper_id(message: str) -> str | None:
    """匹配用户消息提到的第一篇论文（单论文场景）。"""
    hits = _match_paper_ids(message)
    return hits[0] if hits else None


# 全库/跨论文意图关键词：命中且未点名具体论文时，遍历知识库全部论文
_LIB_WIDE_RE = re.compile(
    r"知识库|所有|全部|每篇|各篇|几篇|\d+\s*篇|比较|对比|差异|综述|总结.*论文|论文.*总结"
)


def _is_library_wide(message: str) -> bool:
    """判断是否为跨论文/全库提问（如"分析知识库5篇论文并比较"）。"""
    return bool(_LIB_WIDE_RE.search(message))


def _lib_keywords(top_n: int = 10) -> list[str]:
    """从库内论文标题提取偏好关键词（词频 TopN），供 TOOL 推荐排序使用。

    体现"用户偏好"：知识库内容即用户研究方向的信号。
    """
    from collections import Counter

    stop = {
        "with",
        "from",
        "their",
        "this",
        "that",
        "have",
        "been",
        "via",
        "and",
        "the",
        "for",
        "learning",
        "based",
        "cvpr",
        "paper",
    }
    words: list[str] = []
    for it in _load_registry():
        words += re.findall(r"[a-z0-9]+", (it.get("title") or it["paper_id"]).lower())
    cnt = Counter(w for w in words if len(w) > 3 and w not in stop and not w.isdigit())
    return [w for w, _ in cnt.most_common(top_n)]


def _build_search_query(message: str) -> str | None:
    """中文提问时，用 LLM 把问题改写为英文检索关键词（召回增强）。

    背景：RAG 可能处于两种向量后端——
      - 语义向量（local/remote）：本身支持中英跨语言，但把口语化长问句压成
        论文术语关键词，仍能显著提升召回（尤其缩写、同义词）；
      - 哈希兜底：只有词面匹配能力，不改写则中文提问命中英文语料趋近 0。
    因此统一做一次关键词改写；离线 Mock 模式下无翻译能力，返回 None 退回原查询。
    """
    import re

    if not re.search(r"[一-鿿]", message):
        return None  # 纯英文提问无需改写
    if getattr(_agent, "USE_MOCK_LLM", True):
        return None
    prompt = (
        "把以下科研提问改写为用于英文论文全文检索的英文关键词短语，"
        "只输出 5~15 个英文关键词，不要输出任何解释：\n" + message
    )
    try:
        keywords = _agent.get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
        return keywords or None
    except Exception:
        return None  # 翻译失败不阻断主流程，退回原查询


# 前端「思考强度」控件：标签 + 对应的溯源检索 top_k
#   light  -> 快速直答，检索条数少
#   medium -> 默认行为（与历史版本一致）
#   deep   -> 深度思考，扩大溯源检索范围（更多锚点）
_THINKING_LABEL = {
    "light": "浅度（快速直答）",
    "medium": "中度（默认）",
    "deep": "深度（扩大溯源检索范围）",
}
_THINKING_TOP_K = {"light": 3, "medium": 5, "deep": 8}


@app.get("/api/health")
def health():
    """运行状态：API 版本 + 当前 Embedding 后端 + 向量库规模。

    前端/评审可直接访问该接口确认 RAG 是否已从「本地哈希向量」切换到语义模型，
    无需翻日志（`semantic=false` 即说明仍在哈希兜底）。
    """
    from embeddings import describe  # 延迟导入：RAG 模块未就绪时也能返回基础信息

    info: dict[str, Any] = {
        "backend": "none",
        "model": "",
        "dim": 0,
        "semantic": False,
        "label": "RAG 模块未加载",
    }
    collection, points = "", 0
    if _rag_mod is not None:
        emb = getattr(_rag_mod, "embeddings", None)
        if emb is not None:
            info = describe(emb)
        collection = getattr(_rag_mod, "COLLECTION_NAME", "")
        try:
            points = int(_rag_mod.qdrant.count(collection, exact=True).count)
        except Exception:
            points = 0
    return {
        "ok": True,
        "api_version": API_VERSION,
        "embedding": info,
        "collection": collection,
        "points": points,
    }


@app.get("/")
def index():
    """返回单页前端。

    显式禁用缓存：否则改完 index.html 后浏览器仍用旧页面（只刷新数据不重载 HTML），
    会出现「代码已改、界面没变」的假象。
    """
    return FileResponse(
        os.path.join(BASE_DIR, "index.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/pdf.min.js")
def pdf_js():
    """本地 pdf.js 主库（避免 CDN 不可达导致阅读器不可用）。"""
    return FileResponse(os.path.join(BASE_DIR, "pdf.min.js"), media_type="application/javascript")


@app.get("/pdf.worker.min.js")
def pdf_worker_js():
    """本地 pdf.js Worker 脚本。"""
    return FileResponse(
        os.path.join(BASE_DIR, "pdf.worker.min.js"), media_type="application/javascript"
    )


def _sse(payload: dict[str, Any]) -> str:
    """序列化一条 SSE 消息：`data: <json>\n\n`。

    统一用 JSON 里的 `type` 字段区分事件类型（token / log / final / error），
    前端 fetch reader 按行取 `data:` 前缀后 JSON.parse 即可，比解析 `event:` 字段更简单可靠。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    对话主入口（流式 SSE）：
      1. 以 `astream_events` 驱动 LangGraph，把 LLM 正文逐 token 推送（token 事件）；
      2. 思考日志（路由 / HITL / 分支状态）作为 log 事件实时推送；
      3. 命中 GRAPH 分支触发 HITL 打断时，自动确认后继续流式；
      4. 富数据（图谱 JSON / Base64 图像 / 论文卡片 / 溯源锚点）在流结束时
         由 final 事件一次性返回。
    """
    return StreamingResponse(
        _chat_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 反代场景禁用缓冲，保证逐 token 即时推送
        },
    )


async def _chat_stream(req: ChatRequest):
    """流式对话核心：async generator，逐条 yield SSE 消息。"""
    session_id = req.session_id or str(uuid.uuid4())
    thread = {"configurable": {"thread_id": session_id}}
    log: list[str] = []  # 思考日志：流式推送的同时留档，final 事件里一并返回

    def emit_log(line: str) -> str:
        """追加思考日志并构造对应 log 事件。"""
        log.append(line)
        return _sse({"type": "log", "content": line})

    # ---- 组装工作流输入（与原同步实现一致） ----
    # 论文定位优先级：显式指定 > 消息中点名匹配；匹配到多篇或命中全库意图时
    # 进入多论文遍历模式（Agent 逐篇检索后综合对比）
    matched_ids = req.paper_ids or _match_paper_ids(req.message)
    paper_id = req.paper_id or (matched_ids[0] if len(matched_ids) == 1 else None)
    paper_ids: list[str] = []
    if not paper_id:
        if len(matched_ids) >= 2:
            paper_ids = matched_ids  # 点名多篇：遍历这几篇
        elif _is_library_wide(req.message):
            paper_ids = [it["paper_id"] for it in _load_registry()]  # 全库遍历
    search_query = _build_search_query(req.message)
    # 显式标注 dict[str, object]，避免后续赋 str 值时类型收窄报错
    state_in: dict[str, object] = {"messages": [HumanMessage(content=req.message)]}
    if paper_id:
        state_in["paper_id"] = paper_id
    if paper_ids:
        state_in["paper_ids"] = paper_ids
    if search_query:
        state_in["search_query"] = search_query
    # TOOL 分支上下文：下载保存目录 + 库内偏好关键词（推荐排序用）
    state_in["upload_dir"] = UPLOAD_DIR
    lib_kw = _lib_keywords()
    if lib_kw:
        state_in["lib_keywords"] = lib_kw
    if req.csv_path:
        state_in["csv_path"] = req.csv_path
    # WRITE 分支：注入库内真实文献清单（无幻觉 References 来源）
    state_in["reference_catalog"] = _build_reference_catalog(paper_ids or matched_ids or None)
    # 引用数据优先级：请求显式携带 > 库内论文真实引用网络 > 演示数据兜底
    citations = req.citations_data
    internal_ids: list[str] = []
    cite_source = "请求携带"
    if not citations:
        citations, internal_ids = _build_real_citations()
        cite_source = "库内论文真实引用网络" if citations else "演示数据"
        citations = citations or DEFAULT_CITATIONS
    state_in["citations_data"] = citations

    # ---- 前置思考日志：预处理结果先于 LLM 流式推送 ----
    if req.paper_ids:
        yield emit_log(f"用户已显式限定检索范围（{len(req.paper_ids)} 篇），跳过自动论文匹配。")
    if req.thinking_level:
        yield emit_log(f"思考强度：{_THINKING_LABEL.get(req.thinking_level, req.thinking_level)}")
    if paper_id:
        yield emit_log(f"识别到目标论文 -> {paper_id}，RAG 检索将按此论文过滤。")
    if paper_ids:
        yield emit_log(f"检测到跨论文/全库提问，将逐篇遍历 {len(paper_ids)} 篇论文检索后综合对比。")
    if lib_kw:
        yield emit_log(f"库内偏好关键词：{', '.join(lib_kw[:6])}（供工具推荐排序）。")
    if search_query:
        yield emit_log(f"中文提问已改写为英文检索词 -> {search_query}")

    async def _stream_graph(input_state, ignore_router_token: bool):
        """驱动一次图执行，逐 token 推送非路由节点的 LLM 增量。"""
        async for event in graph_app.astream_events(input_state, config=thread, version="v2"):
            if event.get("event") != "on_chat_model_stream":
                continue
            # router 的意图分类 token 不是最终回答，必须过滤，避免混入正文
            node = (event.get("metadata") or {}).get("langgraph_node")
            if ignore_router_token and node == "router":
                continue
            chunk = (event.get("data") or {}).get("chunk")
            content = getattr(chunk, "content", "") if chunk else ""
            if isinstance(content, str) and content:
                yield _sse({"type": "token", "content": content})

    # ---- 第一次流式执行：GRAPH 分支会在 graph_node 前被 interrupt 暂停 ----
    try:
        async for sse in _stream_graph(state_in, ignore_router_token=True):
            yield sse
    except Exception as e:  # noqa: BLE001 —— 流式失败回传 error 事件，避免连接悬挂
        yield _sse({"type": "error", "message": f"工作流执行失败：{e}"})
        return

    out = graph_app.get_state(thread).values
    branch = out.get("next_step", "RAG")
    yield emit_log(f"Router 意图分类完成 -> {branch}")

    # ---- Human-in-the-Loop：暂停则自动确认恢复（演示确认环节） ----
    snapshot = graph_app.get_state(thread)
    if snapshot.next:
        yield emit_log(f"[HITL] 流程在 {snapshot.next} 前暂停，等待人工确认引用数据...")
        yield emit_log("[HITL] 人工确认无误，恢复执行。")
        try:
            async for sse in _stream_graph(None, ignore_router_token=False):
                yield sse
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"恢复执行失败：{e}"})
            return
        out = graph_app.get_state(thread).values

    answer = out["messages"][-1].content
    yield emit_log(f"{branch} 分支执行完成，结果已返回。")

    # ---- 按分支组装富数据（与原同步实现一致） ----
    resp: dict[str, Any] = {
        "session_id": session_id,
        "branch": branch,
        "answer": answer,
        "log": log,
        "paper_id": paper_id,  # 前端锚点芯片点击跳转原文阅读器时使用
    }
    # PLAN 复合任务：记录子任务序列；含图谱子任务时同样返回图谱 JSON
    plan_meta = out.get("plan_meta") or {}
    if "PLAN" in branch and plan_meta.get("tasks"):
        yield emit_log(f"PLAN 复合任务拆解：{' → '.join(plan_meta['tasks'])}，已依次执行合并。")

    if "GRAPH" in branch or ("PLAN" in branch and plan_meta.get("has_graph")):
        # 图谱 JSON：节点含 PageRank 值与介数中心性，供前端 ECharts 渲染
        yield emit_log(f"引用数据来源：{cite_source}（{len(citations)} 条边）。")
        resp["graph"] = build_and_analyze_citation_graph(citations)
        # 标注库内论文节点，前端分色显示（库内蓝 / 外部灰 / 基石红）
        resp["graph"]["internal_ids"] = internal_ids
    elif "VISUAL" in branch:
        # 模块六可视化：表格 + 折线 / 柱状 / 三维雷达图 + 自适应学术图注
        try:
            # LLM 自适应图注：配置了 Key 时用 LLM 提炼核心趋势，否则退回模板句
            def _caption_fn(summary, chart_type, instruction):
                if getattr(_agent, "USE_MOCK_LLM", True):
                    raise RuntimeError("mock")
                return writing.generate_figure_caption(
                    _agent.get_llm(),
                    writing.build_stats_text(summary),
                    chart_type,
                    instruction,
                )

            viz = execute_data_visualization_v2(
                req.csv_path or SAMPLE_CSV, req.message, caption_fn=_caption_fn
            )
            resp["table"] = viz["table"]
            resp["charts"] = viz["charts"]
            # 向后兼容：首张图仍落到 image/caption 字段
            if viz["charts"]:
                resp["image"] = viz["charts"][0]["image_base64"]
                resp["caption"] = viz["charts"][0]["caption"]
        except Exception as e:
            yield emit_log(f"绘图为可选增强，失败不影响回答：{e}")
    elif "WRITE" in branch:
        # 论文框架：结构化段落 + 无幻觉 References（DOI 补全）
        write_meta = out.get("write_meta") or {}
        refs = _enrich_references(write_meta.get("references") or [])
        refs_text = _format_references_text(refs)
        resp["framework"] = writing._extract_sections(write_meta.get("framework") or answer)
        resp["references"] = refs
        # 把 References 追加进正文气泡（与框架一起展示，含 DOI）
        resp["answer"] = (
            f"{answer}\n\n———\nReferences（绑定知识库真实文献，DOI 经 Crossref 校验）：\n{refs_text}"
        )
        yield emit_log(
            f"论文框架已生成，References 绑定 {len(refs)} 篇库内真实文献"
            f"（{'、'.join(str(r['paper_id'])[:24] for r in refs[:3])}{'…' if len(refs) > 3 else ''}）。"
        )
    elif "DIAGRAM" in branch:
        # 架构拓扑图脚本：多格式代码 + 复制 / Mermaid 预览
        diagram_meta = out.get("diagram_meta") or {}
        resp["diagram"] = diagram_meta.get("scripts") or {}
        yield emit_log(f"拓扑图脚本已生成：{'、'.join(resp['diagram'].keys())}。")
    elif "TOOL" in branch:
        # 工具调用结果：论文卡片数据供前端渲染"一键导入"
        tool = out.get("tool_results") or {}
        resp["tool_papers"] = tool.get("papers", [])
        # download 闭环：tool_node 已下载 PDF，此处自动查重并解析入库
        if tool.get("downloaded_file"):
            msg = _ingest_downloaded_pdf(tool["downloaded_file"], tool.get("downloaded_title"))
            yield emit_log(msg)
            resp["answer"] = f"{answer}\n{msg}"
    else:
        # RAG：附带溯源锚点（页码 + 坐标），供前端高亮定位；
        # 与 rag_node 保持一致的过滤与改写策略，避免锚点指向其他论文。
        # 多论文模式：逐篇取 top3 并保留各自 paper_id，前端锚点按各自论文跳转
        if paper_ids:
            anchors = []
            for pid in paper_ids:
                anchors.extend(
                    retrieve_with_anchors(search_query or req.message, paper_id=pid, top_k=3)
                )
            resp["anchors"] = anchors
            resp["paper_ids"] = paper_ids
        else:
            # 思考强度决定溯源锚点数量（默认 medium = 5，与历史行为一致）
            top_k = _THINKING_TOP_K.get((req.thinking_level or "medium").lower(), 5)
            resp["anchors"] = retrieve_with_anchors(
                search_query or req.message, paper_id=paper_id, top_k=top_k
            )

    # PLAN 复合任务：含图谱子任务时会进入上面的 GRAPH 分支而跳过锚点聚合，
    # 此处独立补齐——RAG 子任务的出处锚点照常返回
    if "PLAN" in branch and paper_ids and "anchors" not in resp:
        anchors = []
        for pid in paper_ids:
            anchors.extend(
                retrieve_with_anchors(search_query or req.message, paper_id=pid, top_k=3)
            )
        resp["anchors"] = anchors
        resp["paper_ids"] = paper_ids

    # ---- 流结束：final 事件携带权威 answer + 富数据 + 完整日志 ----
    yield _sse({**resp, "type": "final"})


@app.get("/api/chat/{session_id}/replay")
def replay_chat(session_id: str):
    """历史对话答案回补：从 LangGraph checkpoint 读取该会话的全部问答对。

    背景：旧版前端历史记录只存了用户提问（回答未入档），恢复时缺回答。
    checkpointer（MemorySaver）按 thread_id 持有每轮的 HumanMessage / AIMessage，
    据此把丢失的回答补回前端；WRITE / DIAGRAM 的结构化产出也在 state 里，一并返回。
    服务重启后 checkpoint 清空，此时返回空列表，前端保持原样不报错。
    """
    thread = {"configurable": {"thread_id": session_id}}
    try:
        values = graph_app.get_state(thread).values or {}
    except Exception:
        values = {}
    msgs_out: list[dict[str, str]] = []
    for m in values.get("messages") or []:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)
        if content:
            msgs_out.append({"role": role, "text": content})
    write_meta = values.get("write_meta") or {}
    diagram_meta = values.get("diagram_meta") or {}
    return {
        "messages": msgs_out,
        "next_step": values.get("next_step", ""),
        "framework": write_meta.get("framework"),
        "references": write_meta.get("references"),
        "diagram": diagram_meta.get("scripts"),
    }


# ============================================================
# 论文知识库：PDF 导入 / 列表 / 删除
# 入库文件保存在 uploads/，元数据登记在 papers.json，
# 向量数据写入 Qdrant（含 page_num / bbox 溯源锚点）。
# ============================================================
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
REGISTRY_FILE = os.path.join(BASE_DIR, "papers.json")


def _load_registry() -> list[dict[str, Any]]:
    """读取论文登记簿（paper_id / 文件名 / 文本块数 / 导入时间）。"""
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_registry(items: list[dict[str, Any]]):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _build_reference_catalog(paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """构建「无幻觉 References」的库内真实文献清单（轻量，不联网）。

    每条含 paper_id / title / authors / journal / year / doi（已有缓存字段）。
    DOI 补全在 WRITE 分支按需执行（见 _enrich_references），避免每次请求都联网。
    """
    papers = _load_registry()
    if paper_ids:
        papers = [p for p in papers if p["paper_id"] in paper_ids]
    catalog: list[dict[str, Any]] = []
    for it in papers:
        catalog.append(
            {
                "paper_id": it["paper_id"],
                "title": (it.get("title") or it.get("paper_id") or "").strip(),
                "authors": it.get("authors") or [],
                "journal": it.get("journal") or "",
                "year": str(it.get("year") or "").strip(),
                "doi": str(it.get("doi") or "").strip(),
            }
        )
    return catalog


def _enrich_references(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为被引文献补全 DOI（Crossref 标题检索），写回登记簿缓存后返回增强结果。

    仅 WRITE 分支调用，且只处理实际被引用的几篇，网络开销可控；失败保持原样（无幻觉，非空穴来风）。
    """
    if not refs:
        return refs
    papers = _load_registry()
    changed = False
    for r in refs:
        if r.get("doi"):
            continue
        title = (r.get("title") or "").strip()
        if not title:
            continue
        try:
            meta = writing.resolve_doi_by_title(title)
        except Exception:
            meta = {}
        if meta.get("doi"):
            r["doi"] = meta["doi"]
            r["authors"] = meta.get("authors") or r.get("authors")
            r["journal"] = meta.get("journal") or r.get("journal")
            r["year"] = meta.get("year") or r.get("year")
            if meta.get("title"):
                r["title"] = meta["title"]
            # 写回登记簿缓存
            for it in papers:
                if it["paper_id"] == r.get("paper_id"):
                    it["doi"] = meta["doi"]
                    it["authors"] = r["authors"]
                    it["journal"] = r["journal"]
                    it["year"] = r["year"]
                    if meta.get("title"):
                        it["title"] = meta["title"]
                    changed = True
                    break
    if changed:
        _save_registry(papers)
    return refs


def _format_references_text(refs: list[dict[str, Any]]) -> str:
    """把文献清单格式化为 References 文本（作者. "标题". 期刊. 年份. DOI）。"""
    if not refs:
        return "（知识库为空，无可引用的真实文献。）"
    lines = []
    for i, r in enumerate(refs, 1):
        authors = ", ".join(r.get("authors") or [])
        title = r.get("title") or r.get("paper_id") or ""
        journal = r.get("journal") or ""
        year = r.get("year") or ""
        doi = r.get("doi") or ""
        meta = f"{authors}. " if authors else ""
        meta += f'"{title}".'
        if journal:
            meta += f" {journal}."
        if year:
            meta += f" {year}."
        if doi:
            meta += f" DOI: {doi}"
        lines.append(f"[{i}] {meta}")
    return "\n".join(lines)


@app.get("/api/papers")
def list_papers():
    """返回已入库论文列表，供前端知识库侧边栏渲染。"""
    return {"papers": _load_registry()}


@app.get("/api/papers/{paper_id}/file")
def get_paper_file(paper_id: str):
    """在线阅读：按 paper_id 回传原始 PDF。

    Content-Disposition 用 inline 使浏览器内嵌展示而非下载；
    前端 iframe 加载时支持 #page=N 片段跳转到指定页。
    """
    for it in _load_registry():
        if it["paper_id"] == paper_id:
            path = os.path.join(UPLOAD_DIR, it["filename"])
            if os.path.exists(path):
                return FileResponse(
                    path,
                    media_type="application/pdf",
                    # filename*=UTF-8'' 形式支持中文/空格文件名
                    headers={
                        "Content-Disposition": (f"inline; filename*=UTF-8''{quote(it['filename'])}")
                    },
                )
            break
    return JSONResponse(status_code=404, content={"error": "文件不存在或未入库"})


# ---- 阅读器数据缓存：按 (文件名, mtime) 失效，避免反复解析/OCR 同一篇 ----
_reader_cache: dict[str, tuple[str, dict[str, Any]]] = {}


def _extract_internal_links(doc) -> list[dict[str, Any]]:
    """提取 PDF 内部跳转（正文 [n] 引用、交叉引用等），返回
    {page, rect, target_page}，前端据此叠加热区实现「引用标记 ↔ 条目」双向跳转。"""
    links: list[dict[str, Any]] = []
    try:
        names = doc.resolve_names()  # name -> 页码(int) 或 (页码, 目标描述)
    except Exception:
        names = {}
    for pno, page in enumerate(doc, start=1):
        for lnk in page.get_links():
            kind = lnk.get("kind")
            target = lnk.get("page")
            if target is None and kind == fitz.LINK_NAMED:
                nd = lnk.get("nameddest")
                val = names.get(nd)
                if isinstance(val, int):
                    target = val
                elif isinstance(val, tuple) and val:
                    target = val[0]
            if target is None or not lnk.get("from"):
                continue
            x0, y0, x1, y1 = lnk["from"]
            links.append(
                {
                    "page": pno,
                    "rect": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                    "target_page": int(target) + 1,
                }
            )
    return links


_SECTION_NAMES = {
    "abstract",
    "references",
    "reference",
    "appendix",
    "acknowledgments",
    "acknowledgements",
    "acknowledgment",
    "acknowledgement",
    "supplementary",
    "supplementary material",
    "conclusion",
    "conclusions",
    "concluding remarks",
    "摘要",
    "参考文献",
    "附录",
    "致谢",
}
_REF_NAMES = {"references", "reference", "参考文献"}
_NUM_HEAD_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\s*[.、]?\s+(\S.*)$")
_ONLY_NUM_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\s*[.、]?$")


def _style_ratio(spans, flag: int) -> float:
    """行内指定样式（span flags 位：16=粗体、2=斜体）的字符占比。"""
    total = sum(len(s.get("text", "")) for s in spans) or 1
    hit = sum(len(s.get("text", "")) for s in spans if (s.get("flags") or 0) & flag)
    return hit / total


def _bold_ratio(spans) -> float:
    """行内粗体字符占比：整行加粗才算标题样式，行内个别粗体词不算。"""
    return _style_ratio(spans, 16)


def _build_heading_toc(doc) -> list[dict[str, Any]]:
    """无 PDF 书签时，从正文推断「书籍式」一/二/三级标题树。

    以结构化信号为主、字号为辅，避免把图注 / 正文粗体片段 / 机构名误判为标题：
      1. 章节编号行（1 / 2.1 / 3.1.1，须为标题样式；一级编号 ≤30 防年份误判）
         → 层级 = 编号深度；仅编号的行（如 "2.2."）标记 pending 等待合并续行；
      2. 常见无编号节名（Abstract / References / 附录 等，整行匹配）→ 一级；
      3. 字号显著大于正文（≥ +2.5pt）的行 → 一级（论文大标题），可跨行合并；
      4. 其余标题样式行只作断行标题的「续行素材」（fill），不直接进目录；
      5. 过滤 URL / 参考文献条目 / 机构地址等噪声；References 区之后不再产标题。
    """
    import statistics

    rows = []  # (page, col, y, x0, size, bold, italic, text) 按收集顺序（页序 + 块序 ≈ 阅读序）
    for pno, page in enumerate(doc, start=1):
        mid = page.rect.width / 2
        try:
            d = page.get_text("dict", flags=0)
        except Exception:
            continue
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = " ".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                x0, y0 = line["bbox"][0], line["bbox"][1]
                size = max(s["size"] for s in spans)
                bold = _bold_ratio(spans) > 0.6
                italic = _style_ratio(spans, 2) > 0.6  # 整行斜体：机构名 / 图注等
                # 行内混合字号（差 > 0.5pt）= 含上标/下标（作者机构上标编号、公式等），非标题
                mixed = (max(s["size"] for s in spans) - min(s["size"] for s in spans)) > 0.5
                rows.append(
                    (
                        pno,
                        0 if x0 < mid else 1,
                        round(y0, 1),
                        round(x0, 1),
                        round(size, 1),
                        bold,
                        italic,
                        mixed,
                        text,
                    )
                )
    if not rows:
        return []
    try:
        body = statistics.mode([r[4] for r in rows])
    except statistics.StatisticsError:
        body = statistics.median([r[4] for r in rows])

    def noisy(t: str) -> bool:
        if len(t) < 2 or len(t) > 95:
            return True
        if "http" in t or "www." in t or "@" in t:
            return True
        low = t.lower()
        if low.startswith("arxiv:") or low.startswith("arxiv "):
            return True  # arXiv 侧边水印
        return t.startswith("[")

    cand: list[dict[str, Any]] = []
    in_refs = False
    for page, col, y, x0, size, bold, italic, mixed, text in rows:
        t = text.strip()
        key = t.lower().rstrip(":.,；。 ")
        if key in _SECTION_NAMES:
            cand.append(
                {
                    "page": page,
                    "col": col,
                    "_y": y,
                    "x0": x0,
                    "size": size,
                    "level": 1,
                    "title": t.rstrip(":.,；。 "),
                    "join": True,
                }
            )
            if key in _REF_NAMES:
                in_refs = True  # References 区之后不再产标题
            continue
        if in_refs or noisy(t) or italic:
            continue
        if not (bold or size >= body + 0.6):  # 必须是标题样式（整行加粗或字号更大）
            continue
        m = _NUM_HEAD_RE.match(t)
        only = _ONLY_NUM_RE.match(t)
        if only:  # 仅编号（如 "2.2."）：标题断行，合并续行
            first = int(only.group(1).split(".")[0])
            if 1 <= first <= 30 and not mixed:
                cand.append(
                    {
                        "page": page,
                        "col": col,
                        "_y": y,
                        "x0": x0,
                        "size": size,
                        "level": min(only.group(1).count(".") + 1, 3),
                        "title": t,
                        "join": True,
                    }
                )
        elif m:
            first = int(m.group(1).split(".")[0])
            if 1 <= first <= 30 and re.search(r"[A-Za-z]", m.group(2)) and not mixed:
                # 编号后须含字母（排除表格数字 "8.1 ×"）、行内无上下标混排（排除机构上标编号）
                cand.append(
                    {
                        "page": page,
                        "col": col,
                        "_y": y,
                        "x0": x0,
                        "size": size,
                        "level": min(m.group(1).count(".") + 1, 3),
                        "title": t,
                        "join": True,
                    }
                )
        elif (
            bold
            and size >= body + 1.8
            and len(t) <= 90
            and len(t) >= 8
            and re.search(r"[A-Za-z]{2,}", t)
            and not mixed
        ):
            cand.append(
                {
                    "page": page,
                    "col": col,
                    "_y": y,
                    "x0": x0,
                    "size": size,
                    "level": 1,
                    "title": t,
                    "big": True,
                    "join": True,
                }
            )  # 论文大标题 / 大号节名
        else:
            # 标题样式但非编号/白名单/大字号：仅作续行素材，不直接进目录
            cand.append(
                {
                    "page": page,
                    "col": col,
                    "_y": y,
                    "x0": x0,
                    "size": size,
                    "title": t,
                    "fill": True,
                }
            )
    if not cand:
        return []

    # 断行标题合并：标题行（join）吸收紧邻的续行素材（fill）；大标题可吸收紧邻的另一段大标题行。
    # 同 y 时 join 行优先（"2.1" 与标题文字同行拆分为两个 line 的排版）；
    # dy 允许 =0（同行拆分）也可跨行（断行标题）。
    cand.sort(key=lambda c: (c["page"], c["col"], c["_y"], 0 if c.get("join") else 1))
    merged: list[dict[str, Any]] = []
    for c in cand:
        prev = merged[-1] if merged else None
        joinable = c.get("fill") or (prev is not None and prev.get("big") and c.get("big"))
        if prev and prev.get("join") and joinable:
            x0_tol = 200 if prev.get("big") else 60
            same = (
                abs(prev["size"] - c["size"]) < 0.35
                and prev["col"] == c["col"]
                and abs(prev["x0"] - c["x0"]) < x0_tol
            )
            near = c["page"] == prev["page"] and 0 <= c["_y"] - prev["_y"] < prev["size"] * 1.9
            big_ok = not prev.get("big") or c["size"] >= body + 1.8
            if same and near and big_ok and len(prev["title"]) + len(c["title"]) < 120:
                prev["title"] = (prev["title"] + " " + c["title"]).strip()
                prev["_y"] = c["_y"]  # 续行合并后以最新行继续判断行距
                continue
        if not c.get("fill"):
            merged.append(c)

    # 仅编号且未合并到任何续行的条目 = 表格数字 / 断行残片，丢弃
    return [
        {"level": c["level"], "title": re.sub(r"\s+", " ", c["title"]).strip(), "page": c["page"]}
        for c in merged
        if not _ONLY_NUM_RE.match(c["title"])
    ]


_MATH_TOKEN_RE = re.compile(r"[=+\-*/^_{}()\[\]<>≤≥≠≈±×÷∑∏∫√∂∇∞∈∉⊂⊆→⇒↔|°′″]|[α-ωΑ-Ω]|\\[A-Za-z]+")


# ---------------------------------------------------------------------------
# 公式 LaTeX 还原（规则层）：把 PDF 文本层里的 Unicode 数学记号转成 LaTeX。
# 零依赖、离线、即时，覆盖学术论文最常用的记号；无法映射的字符原样保留，
# 输出永远可用。视觉级精修（pix2tex）见下方 `_get_latex_ocr`，按需回退到此层。
# ---------------------------------------------------------------------------
_GREEK_MAP = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
}
_SYM_MAP = {
    "×": r"\times",
    "÷": r"\div",
    "±": r"\pm",
    "∓": r"\mp",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "≡": r"\equiv",
    "∼": r"\sim",
    "∝": r"\propto",
    "∞": r"\infty",
    "∂": r"\partial",
    "∇": r"\nabla",
    "⋅": r"\cdot",
    "∘": r"\circ",
    "⊕": r"\oplus",
    "⊗": r"\otimes",
    "⊥": r"\perp",
    "∥": r"\parallel",
    "∠": r"\angle",
    "∈": r"\in",
    "∉": r"\notin",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "⊃": r"\supset",
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
    "∀": r"\forall",
    "∃": r"\exists",
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "↔": r"\leftrightarrow",
    "⇌": r"\rightleftharpoons",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∬": r"\iint",
    "∮": r"\oint",
    "⟨": r"\langle ",
    "⟩": r" \rangle",
    "ℓ": r"\ell",
    "ℝ": r"\mathbb{R}",
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}",
    "ℂ": r"\mathbb{C}",
    "…": r"\ldots",
    "⋯": r"\cdots",
    "−": "-",
    "·": r"\cdot ",
    "°": r"^{\circ}",
}
_SUPER_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁿ": "n",
    "ⁱ": "i",
    "ᵀ": "T",
    "ᴺ": "N",
    "ᵏ": "k",
    "ᵐ": "m",
}
_SUB_MAP = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
    "ₐ": "a",
    "ₑ": "e",
    "ₕ": "h",
    "ᵢ": "i",
    "ⱼ": "j",
    "ₖ": "k",
    "ₗ": "l",
    "ₘ": "m",
    "ₙ": "n",
    "ₒ": "o",
    "ₚ": "p",
    "ᵣ": "r",
    "ₛ": "s",
    "ₜ": "t",
    "ᵤ": "u",
    "ᵥ": "v",
    "ₓ": "x",
}
_SQRT_TAIL_RE = re.compile(r"[A-Za-z0-9]+")


def _cmd(v: str) -> str:
    """命令映射值统一带尾空格，防止 \alpha + x 拼成 \alphax。"""
    return v + " " if v and v[-1].isalpha() and not v.endswith(" ") else v


def _unicode_to_latex(text: str) -> str:
    """Unicode 数学记号 → LaTeX。

    先修复被 PDF 拆散的 LaTeX 命令（\\ma t hbf → \\mathbf）；
    连续上/下标字符合并为 ^{} / _{}；√ 后跟字母数字时写成 \\sqrt{...}；
    LaTeX 特殊字符（% &）转义；未映射字符原样保留。
    """
    text = _fix_latex_spacing(text)
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _SUPER_MAP or ch in _SUB_MAP:
            is_sup = ch in _SUPER_MAP
            tbl = _SUPER_MAP if is_sup else _SUB_MAP
            buf: list[str] = []
            while i < n and text[i] in tbl:
                buf.append(tbl[text[i]])
                i += 1
            body = "".join(buf)
            mark = "^" if is_sup else "_"
            out.append(mark + body if len(body) == 1 else mark + "{%s}" % body)
            continue
        if ch == "√":
            i += 1
            m = _SQRT_TAIL_RE.match(text, i)
            radicand = text[i : m.end()] if m else ""
            i = m.end() if m else i
            out.append(r"\sqrt{%s}" % radicand)
            continue
        if ch in _GREEK_MAP:
            out.append(_cmd(_GREEK_MAP[ch]))
        elif ch in _SYM_MAP:
            out.append(_cmd(_SYM_MAP[ch]))
        elif ch == "%":
            out.append(r"\%")
        elif ch == "&":
            out.append(r"\&")
        else:
            out.append(ch)
        i += 1
    s = "".join(out)

    # 回收命令后不必要的尾空格：\sum _{i=1} -> \sum_{i=1}（仅当下一个字符
    # 不是字母/数字时才去掉；是字母时必须保留，否则又粘连）。
    # 注意不能用 lookahead+回溯写法：\sum_ 会被回溯成 \su + m，把命令名拆坏。
    def _trim(m: "re.Match[str]") -> str:
        nxt = s[m.end() : m.end() + 1]
        return m.group(0)[:-1] + (" " if nxt and re.match(r"[A-Za-z0-9]", nxt) else "")

    s = re.sub(r"\\[A-Za-z]+ ", _trim, s)
    return re.sub(r" {2,}", " ", s).strip()


_EQ_NO_RE = re.compile(r"\((\d+(?:\.\d+)*)\)\s*$")

# PDF 文本层常把 LaTeX 命令按字距拆散（\mathbf 提取成 \ma t hbf）。
# 仅当拼接结果命中已知命令表时才合并，避免误伤合法的字母间空格。
_LATEX_CMD_NAMES = {
    "mathbf",
    "mathcal",
    "mathrm",
    "mathit",
    "mathbb",
    "mathsf",
    "mathtt",
    "boldsymbol",
    "operatorname",
    "frac",
    "dfrac",
    "tfrac",
    "text",
    "sqrt",
    "sum",
    "prod",
    "int",
    "oint",
    "left",
    "right",
    "cdot",
    "times",
    "div",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "varsigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Phi",
    "Psi",
    "Omega",
    "infty",
    "partial",
    "nabla",
    "pm",
    "mp",
    "leq",
    "geq",
    "neq",
    "approx",
    "equiv",
    "sim",
    "propto",
    "subset",
    "subseteq",
    "supset",
    "in",
    "notin",
    "cup",
    "cap",
    "emptyset",
    "forall",
    "exists",
    "rightarrow",
    "leftarrow",
    "Rightarrow",
    "Leftarrow",
    "leftrightarrow",
    "rightarrow",
    "cdot",
    "circ",
    "oplus",
    "otimes",
    "perp",
    "parallel",
    "angle",
    "prime",
    "vec",
    "hat",
    "tilde",
    "bar",
    "dot",
    "overline",
    "underline",
    "widehat",
    "widetilde",
    "quad",
    "qquad",
    "begin",
    "end",
    "item",
    "label",
    "ref",
    "cite",
    "arg",
    "max",
    "min",
    "log",
    "ln",
    "exp",
    "sin",
    "cos",
    "tan",
    "det",
    "dim",
    "rank",
    "tr",
}
_FIX_CMD_SPACING_RE = re.compile(r"\\(?:[A-Za-z]{1,4} )+[A-Za-z]+")


def _fix_latex_spacing(text: str) -> str:
    """修复被 PDF 文本层按字距拆散的 LaTeX 命令（\\ma t hbf → \\mathbf）。"""

    def _join(m: "re.Match[str]") -> str:
        joined = "\\" + m.group(0)[1:].replace(" ", "")
        # 去掉拼接串尾随的非命令部分：只验证前缀是否为已知命令
        frag = m.group(0).replace(" ", "")[1:]
        for k in range(len(frag), 2, -1):
            if frag[:k] in _LATEX_CMD_NAMES:
                return "\\" + frag[:k] + " " + frag[k:]
        return m.group(0)

    return _FIX_CMD_SPACING_RE.sub(_join, text) if "\\" in text else text


def _detect_formulas(doc, toc: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """启发式识别 PDF 文本层中的数学公式块，并做轻量版面语义标注。

    识别特征：文本短、数学符号密度高（每 8 个字符约 1 个数学记号）、
    且不像完整英文句子；排除参考文献条目与超长正文段落。
    每条额外产出（对应赛题「公式完整 LaTeX 还原 + 版面级语义理解」）：
      - latex / latex_source：规则层 LaTeX 还原结果（source="rule"；
        视觉级 pix2tex 精修走 /api/papers/{id}/formula-latex，按需触发）；
      - eq_no：右侧公式编号（如 "(2.3)" → "2.3"，转换前先剥离）；
      - kind：display（纯符号行）/ mixed（少量自然语言词）；
      - section：所属章节标题（书签/推断标题树中不晚于该页的最近条目）；
      - prev/next：版面上相邻正文块的语义上下文片段。
    返回 [{page, bbox, text, ...}]；扫描件（无文本层）返回空列表。
    """
    sec_pages = [(t.get("page", 0), t.get("title", "")) for t in (toc or [])]

    def _sec_of(pno: int) -> str:
        best = ""
        for pg, title in sec_pages:
            if pg <= pno:
                best = title
            else:
                break
        return best[:60]

    formulas: list[dict[str, Any]] = []
    for pno, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        for block in blocks:
            text = " ".join(block[4].split()).strip()
            if len(text) < 3 or len(text) > 240:
                continue
            if re.match(r"^\[\d+\]", text):
                continue
            hits = _MATH_TOKEN_RE.findall(text)
            density = len(hits) / max(len(text), 1)
            words = re.findall(r"[A-Za-z]{2,}", text)
            # 密度阈值 + 词数上限：区分「公式」与「含公式符号的句子」
            if density >= 0.10 and len(words) <= 6:
                by0, by1 = block[1], block[3]
                eq_no = ""
                body = text
                m = _EQ_NO_RE.search(text)
                if m and len(m.group(1)) <= 6:
                    eq_no = m.group(1)
                    body = text[: m.start()].strip()
                # 版面语义：同页 y 向最近的前/后正文块片段
                prev_ctx = next_ctx = ""
                for b in blocks:
                    if b is block or not str(b[4]).strip():
                        continue
                    if b[3] <= by0 and (not prev_ctx or b[3] > prev_ctx[0]):
                        prev_ctx = (b[3], " ".join(str(b[4]).split())[:90])
                    elif b[1] >= by1 and (not next_ctx or b[1] < next_ctx[0]):
                        next_ctx = (b[1], " ".join(str(b[4]).split())[:90])
                formulas.append(
                    {
                        "page": pno,
                        "bbox": [round(v, 2) for v in block[:4]],
                        "text": text,
                        "image": _render_region_png(page, block[:4]),
                        "latex": _unicode_to_latex(body),
                        "latex_source": "rule",
                        "eq_no": eq_no,
                        "kind": "mixed" if words else "display",
                        "section": _sec_of(pno),
                        "prev": prev_ctx[1] if prev_ctx else "",
                        "next": next_ctx[1] if next_ctx else "",
                    }
                )
    return formulas


# ---------------------------------------------------------------------------
# 公式 LaTeX 还原（视觉层，可选）：pix2tex / LaTeX-OCR。
# 依赖 torch（CPU 版即可，约数百 MB），未安装或初始化失败时自动回退规则层，
# 不影响任何既有功能。单条推理 CPU 1–5 秒，因此只做按需端点，不做全量预解析。
# ---------------------------------------------------------------------------
_LATEX_OCR_STATE: dict[str, Any] = {"engine": None, "failed": False}


def _get_latex_ocr() -> Any:
    """懒加载 pix2tex；失败信息记录到状态字典，后续调用快速跳过。"""
    if getattr(config, "ENABLE_LATEX_OCR", "auto") == "off":
        raise RuntimeError("ENABLE_LATEX_OCR=off，已禁用视觉还原")
    if _LATEX_OCR_STATE["failed"]:
        raise RuntimeError("pix2tex 初始化曾失败（缺依赖或权重下载失败）")
    if _LATEX_OCR_STATE["engine"] is None:
        try:
            # 权重托管在 HuggingFace，国内直连常超时；默认走镜像
            # （已显式设置 HF_ENDPOINT 的用户不受影响）
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from pix2tex.cli import LatexOCR  # pyright: ignore[reportMissingImports]

            _LATEX_OCR_STATE["engine"] = LatexOCR()  # 首次会下载约 100MB 权重并初始化模型
        except Exception as e:  # noqa: BLE001
            _LATEX_OCR_STATE["failed"] = True
            raise RuntimeError(f"pix2tex 不可用：{e}") from e
    return _LATEX_OCR_STATE["engine"]


class FormulaLatexRequest(BaseModel):
    """单条公式的视觉 LaTeX 还原请求。"""

    page: int
    bbox: list[float]
    text: str = ""


def _match_caption(blocks, fy0, fy1) -> str:
    """在图片块下方就近寻找图注：y 起点紧贴图片底部（0~30pt）。"""
    best = ""
    for b in blocks:
        if b.get("type") != 0:
            continue
        bx = b["bbox"]
        text = " ".join(
            s["text"] for line in b.get("lines", []) for s in line.get("spans", [])
        ).strip()
        if not text:
            continue
        if bx[1] < fy1 - 2 or bx[1] > fy1 + 30:
            continue
        if re.match(r"^(Fig|Figure|Table|表|图)\b", text, re.IGNORECASE):
            return text
        if not best:
            best = text
    return best


def _render_region_png(page, bbox, dpi: int = 120) -> str:
    """把页面某区域渲染为 PNG（base64），供前端侧栏预览「结构化抽取到了什么」。"""
    try:
        pm = page.get_pixmap(clip=fitz.Rect(bbox), dpi=dpi)
        return base64.b64encode(pm.tobytes("png")).decode()
    except Exception:
        return ""


def _bbox_overlaps(bbox: tuple[float, float, float, float], boxes) -> bool:
    """判断 bbox 是否与 boxes 中任一矩形有显著重叠（面积交集 / 自身面积 > 0.3）。"""
    ax0, ay0, ax1, ay1 = bbox
    area = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    for b in boxes:
        bx0, by0, bx1, by1 = b
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        if ix * iy / area > 0.3:
            return True
    return False


def _rects_close(a, b, gap: float = 15.0) -> bool:
    """两矩形重叠或相邻（间距 < gap），用于聚合矢量图形碎片。"""
    return not (a[2] < b[0] - gap or b[2] < a[0] - gap or a[3] < b[1] - gap or b[3] < a[1] - gap)


def _words_to_cells(words) -> list[list[str]]:
    """把区域内 words 按 y 聚类成行、按 x 聚类成列，还原单元格矩阵。

    words 元素为 get_text("words") 返回的 (x0, y0, x1, y1, text, ...)。
    列切分用「x0 间隙 > 14pt」判定；若切出的列数过多（>8）则回退为单列整行文本。
    """
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    # 行聚类：y0 差 < 8pt 视为同一行
    rows: list[list] = []
    cur = [words[0]]
    for w in words[1:]:
        if w[1] - cur[0][1] < 8:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
    rows.append(cur)
    # 列聚类：收集 x0，间隙 > 14pt 断开
    xs = sorted({round(w[0], 1) for r in rows for w in r})
    cols: list[list[float]] = []
    for x in xs:
        if not cols or x - cols[-1][-1] > 14:
            cols.append([x])
        else:
            cols[-1].append(x)
    if len(cols) > 8:  # 切太碎，回退单列
        return [[(" ".join(w[4] for w in r)).strip()] for r in rows]
    centers = [sum(c) / len(c) for c in cols]
    grid: list[list[str]] = []
    for r in rows:
        cells = {i: "" for i in range(len(centers))}
        for w in r:
            ci = min(range(len(centers)), key=lambda i: abs(w[0] - centers[i]))
            cells[ci] = (cells[ci] + " " + w[4]).strip()
        grid.append([cells[i] for i in range(len(centers))])
    return grid


def _detect_ruleless_tables(
    page, pno, skip
) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    """用 get_drawings 的水平线识别「三线表」——find_tables 漏检的无框线表格。

    规则：top/bottom 两条横线 x 范围基本一致、高度 18~400pt，区域内含 ≥2 行文本，
    文本按 _words_to_cells 还原为单元格矩阵。返回 [(bbox, rows)]。
    """
    hlines: list[tuple[float, float, float]] = []
    try:
        for d in page.get_drawings():
            r = d["rect"]
            if r.height < 2 and r.width > 80:  # 水平线
                hlines.append((r.y0, r.x0, r.x1))
    except Exception:
        return []
    if len(hlines) < 2:
        return []
    hlines.sort()
    merged: list[list] = []
    for y, x0, x1 in hlines:
        if merged and abs(merged[-1][0] - y) < 3:
            merged[-1] = [y, min(merged[-1][1], x0), max(merged[-1][2], x1)]
        else:
            merged.append([y, x0, x1])
    out: list = []
    used = list(skip)  # 内部去重：同一表格可能被表头线/底线多对组合重复命中
    for i in range(len(merged)):
        y0, x0a, x1a = merged[i]
        for j in range(i + 1, len(merged)):
            y1, x0b, x1b = merged[j]
            h = y1 - y0
            if h < 18 or h > 400:
                continue
            x0, x1 = max(x0a, x0b), min(x1a, x1b)
            if x1 - x0 < 80:
                continue
            # 两条线宽度需基本一致（否则是页眉横线等噪声）
            wa, wb = x1a - x0a, x1b - x0b
            if wa < 0.7 * wb or wb < 0.7 * wa:
                continue
            bbox = (x0, y0, x1, y1)
            try:
                words = page.get_text("words", clip=fitz.Rect(bbox))
            except Exception:
                words = []
            if len(words) < 4:
                continue
            rows = _words_to_cells(words)
            if len(rows) < 2:
                continue
            if _bbox_overlaps(bbox, used):
                continue
            used.append(bbox)
            out.append((bbox, rows))
            break
    return out


def _detect_vector_figures(page, pno, existing) -> list[tuple[float, float, float, float]]:
    """用 get_drawings 聚合矢量图形区域，识别非位图的图表（figure）。

    过滤细线（表格线/下划线）、过小图标与超大整页元素，合并重叠矩形，
    排除已识别图片块，返回 figure bbox 列表。
    """
    rects: list[tuple[float, float, float, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    for d in drawings:
        r = d["rect"]
        w, h = r.width, r.height
        if (w > 40 and h < 5) or (h > 40 and w < 5):  # 细线
            continue
        if w < 30 or h < 30:
            continue
        if w > 600 or h > 600:
            continue
        rects.append((r.x0, r.y0, r.x1, r.y1))
    if not rects:
        return []
    merged: list[tuple[float, float, float, float]] = []
    for r in sorted(rects):
        placed = False
        for i, m in enumerate(merged):
            if _rects_close(r, m):
                merged[i] = (min(r[0], m[0]), min(r[1], m[1]), max(r[2], m[2]), max(r[3], m[3]))
                placed = True
                break
        if not placed:
            merged.append(r)
    out = []
    for m in merged:
        if (m[2] - m[0]) < 80 or (m[3] - m[1]) < 60:
            continue
        if _bbox_overlaps(m, existing):
            continue
        out.append(m)
    return out


def _extract_tables_and_figures(doc) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """结构化抽取 PDF 中的表格与图表（figure），并渲染区域缩略图预览。

    表格：`page.find_tables()`（有框线表格）+ `_detect_ruleless_tables`（三线表）；
    图表：图片块（位图）+ `_detect_vector_figures`（矢量图），均向下匹配图注 caption。
    每项附带 `image`（PNG base64）供前端「表格图表」侧栏预览。扫描件无文本层时为空。
    """
    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    for pno, page in enumerate(doc, start=1):
        found_boxes: list[tuple[float, float, float, float]] = []
        # ---- 1. 有框线表格 ----
        try:
            finder = page.find_tables()
        except Exception:
            finder = None
        if finder is not None:
            for t in finder.tables:
                try:
                    rows = t.extract()
                except Exception:
                    continue
                if not rows:
                    continue
                flat = [c or "" for r in rows for c in r]
                if not any(x.strip() for x in flat):
                    continue
                bbox = tuple(t.bbox)
                found_boxes.append(bbox)
                tables.append(
                    {
                        "page": pno,
                        "bbox": [round(v, 2) for v in bbox],
                        "rows": [[c or "" for c in r] for r in rows],
                        "image": _render_region_png(page, bbox),
                    }
                )
        # ---- 2. 三线表（无框线，启发式检测，结构可能切分不准） ----
        for bbox, rows in _detect_ruleless_tables(page, pno, found_boxes):
            found_boxes.append(bbox)
            tables.append(
                {
                    "page": pno,
                    "bbox": [round(v, 2) for v in bbox],
                    "rows": rows,
                    "image": _render_region_png(page, bbox),
                    "ruleless": True,  # 标记来源：整篇翻译只导出可靠的有框线表格
                }
            )
        # ---- 3. 位图图表（图片块） ----
        try:
            blocks = page.get_text("dict").get("blocks", [])
        except Exception:
            blocks = []
        for b in blocks:
            if b.get("type") != 1:
                continue
            x0, y0, x1, y1 = b["bbox"]
            if (x1 - x0) < 40 or (y1 - y0) < 40:  # 过滤小图标 / logo
                continue
            caption = _match_caption(blocks, y0, y1)
            bbox = tuple(b["bbox"])
            figures.append(
                {
                    "page": pno,
                    "bbox": [round(v, 2) for v in bbox],
                    "caption": caption,
                    "image": _render_region_png(page, bbox),
                }
            )
        # ---- 4. 矢量图（非位图图表） ----
        img_boxes = [tuple(f["bbox"]) for f in figures if f["page"] == pno]
        for vb in _detect_vector_figures(page, pno, img_boxes):
            caption = _match_caption(blocks, vb[1], vb[3])
            figures.append(
                {
                    "page": pno,
                    "bbox": [round(v, 2) for v in vb],
                    "caption": caption,
                    "image": _render_region_png(page, vb),
                }
            )
    return tables, figures


@app.get("/api/papers/{paper_id}/reader")
def get_paper_reader(paper_id: str):
    """阅读器数据：目录树 + 文本块（含 page_num/bbox）+ 内部引用链接 + 总页数。

    前端「中英对照翻译」「标题树导航」「引用跳转」三个功能都从这里取数；
    扫描件走同一套 extract_chunks_from_pdf（含 OCR），因此三功能对扫描件同样生效。
    """
    item = next((it for it in _load_registry() if it["paper_id"] == paper_id), None)
    if not item:
        return JSONResponse(status_code=404, content={"error": "论文不存在"})
    path = os.path.join(UPLOAD_DIR, item["filename"])
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "文件不存在或未入库"})
    try:
        mtime = str(os.path.getmtime(path))
    except OSError:
        mtime = ""
    cached = _reader_cache.get(paper_id)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        # min_len 用小阈值：论文标题（如 "Attention Is All You Need"）、章节标题等
        # 短块也要进翻译面板；入库检索仍用默认 30，互不影响。
        chunks = _rag_mod.extract_chunks_from_pdf(path, min_len=4)
    except RuntimeError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    total_pages = 0
    toc: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    formula_keys: set[tuple[Any, ...]] = set()
    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    try:
        doc = fitz.open(path)
        total_pages = doc.page_count
        toc = [{"level": t[0], "title": t[1], "page": t[2]} for t in doc.get_toc(simple=False)]
        if not toc:
            toc = _build_heading_toc(doc)  # 无书签：按字号/粗体推断标题层级
        formulas = _detect_formulas(doc, toc=toc)
        formula_keys = {(f["page"], tuple(f["bbox"])) for f in formulas}
        tables, figures = _extract_tables_and_figures(doc)
        links = _extract_internal_links(doc)
        doc.close()
    except Exception:
        total_pages = max((c["page_num"] for c in chunks), default=0)
    data = {
        "total_pages": total_pages,
        "toc": toc,
        "links": links,
        "formulas": formulas,
        "tables": tables,
        "figures": figures,
        "chunks": [
            {
                "page_num": c["page_num"],
                "bbox": c["bbox"],
                "text": c["text"],
                "is_ref": bool(c.get("is_ref")),
                "is_formula": (c["page_num"], tuple(c["bbox"])) in formula_keys,
            }
            for c in chunks
        ],
    }
    _reader_cache[paper_id] = (mtime, data)
    return data


@app.post("/api/papers/{paper_id}/formula-latex")
def formula_latex(paper_id: str, req: FormulaLatexRequest):
    """单条公式的 LaTeX 还原：优先 pix2tex 视觉识别，不可用则回退规则层。

    规则层结果已随 /reader 下发（latex_source="rule"）；本端点是「AI 精修」按钮
    的后端：对公式区域渲染图跑 LaTeX-OCR，还原上下标/分式等规则层难覆盖的结构。
    pix2tex CPU 单条 1–5 秒，前端按需逐条调用；未安装/失败时返回
    latex_source="rule"（HTTP 200），前端照常展示规则层结果，不报错。
    """
    item = next((it for it in _load_registry() if it["paper_id"] == paper_id), None)
    if not item:
        return JSONResponse(status_code=404, content={"error": "论文不存在"})
    path = os.path.join(UPLOAD_DIR, item["filename"])
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "文件不存在或未入库"})

    fallback = {
        "latex": _unicode_to_latex(_EQ_NO_RE.sub("", req.text).strip()),
        "source": "rule",
        "engine": "unicode-rule",
    }
    try:
        engine = _get_latex_ocr()
    except RuntimeError as e:
        return {**fallback, "note": str(e)}

    try:
        import io
        from PIL import Image  # pyright: ignore[reportMissingImports]

        with fitz.open(path) as doc:
            page = doc[max(0, min(int(req.page) - 1, doc.page_count - 1))]
            bbox = [float(v) for v in (req.bbox or [])]
            if len(bbox) != 4:
                return {**fallback, "note": "bbox 非法"}
            png = _render_region_png(page, bbox, dpi=200)
        if not png:
            return {**fallback, "note": "区域渲染失败"}
        latex = engine(Image.open(io.BytesIO(base64.b64decode(png)))).strip()
        # 模型对宽图会输出一长串填充间隔符（\;\;\;...），截断
        latex = re.sub(r"(?:\\;|\\,|\\quad|\\qquad){3,}.*$", "", latex).strip()
        if not latex:
            return {**fallback, "note": "视觉识别结果为空"}
        return {"latex": latex, "source": "pix2tex", "engine": "LaTeX-OCR (pix2tex)"}
    except Exception as e:  # noqa: BLE001 - 模型/渲染任何异常都回退规则层
        return {**fallback, "note": f"视觉还原失败：{e}"}


def _split_translation(resp: str, n: int) -> list[str]:
    """把 LLM 返回的翻译文本切回 n 段：去编号前缀、按行切、不足补空。"""
    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    cleaned = [re.sub(r"^\s*[【\[]?\s*段落\s*\d+\s*[】\]]?\s*[：:]?\s*", "", ln) for ln in lines]
    if len(cleaned) >= n:
        return cleaned[:n]
    if len(cleaned) == 1 and n > 1:
        return [cleaned[0]] + [""] * (n - 1)
    while len(cleaned) < n:
        cleaned.append("")
    return cleaned[:n]


# ============================================================
# 整篇翻译（原位排版 + Markdown 导出）——模块三最后一块
# ============================================================
def _translate_full_blocks(texts: list[str], target_lang: str) -> list[str]:
    """整篇翻译的批量翻译：按块翻译，公式/代码/引用编号保留原文；返回与输入等长列表。"""
    n = len(texts)
    if getattr(_agent, "USE_MOCK_LLM", True):
        return ["（未配置 LLM，无法翻译）" if t.strip() else "" for t in texts]
    if target_lang == "en":
        head = (
            "把下面的中文段落逐段翻译成符合学术规范的英文，用于论文整篇对照阅读。要求：\n"
            "1. 只输出译文，每段一行，不要编号、不要解释、不要输出原文；\n"
            "2. 公式、LaTeX、数字、变量名、代码、参考文献编号如 [12] 一律保留原文不译；\n"
            "3. 术语准确、语气客观、句式地道，避免中式英语；\n"
            "4. 输出段落数量必须与输入完全一致。\n\n"
        )
    else:
        head = (
            "把下面的英文段落逐段翻译成中文，用于学术论文整篇对照阅读。要求：\n"
            "1. 只输出译文，每段一行，不要编号、不要解释、不要输出原文；\n"
            "2. 公式、LaTeX、数字、变量名、代码、参考文献编号如 [12] 一律保留原文不译；\n"
            "3. 术语准确：attention=注意力、embedding=嵌入、token/transformer 保留英文；\n"
            "4. 输出段落数量必须与输入完全一致。\n\n"
        )
    batch_size = 8
    result: list[str] = [""] * n
    nonempty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    for k in range(0, len(nonempty), batch_size):
        batch = nonempty[k : k + batch_size]
        prompt = head + "\n".join(f"【段落{j + 1}】{t}" for j, (_, t) in enumerate(batch))
        try:
            resp = _agent.get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
            seg = _split_translation(resp, len(batch))
        except Exception as e:  # noqa: BLE001
            seg = [f"（翻译失败：{str(e)[:40]}）"] * len(batch)
        for (i, _t), s in zip(batch, seg):
            result[i] = s
    return result


def _match_heading_level(text: str, toc_norm: list[tuple[str, int, int]], page: int) -> int:
    """判断文本块是否为标题：优先精确/前缀匹配目录标题，再走启发式（编号章节/常见章节名）。"""
    norm = " ".join(text.split()).strip().lower().rstrip(".")
    if not norm:
        return 0
    for t, level, p in toc_norm:
        if not t:
            continue
        if norm == t or norm.startswith(t) or t.startswith(norm):
            if abs(p - page) <= 1:
                return level
    s = text.strip()
    if len(s) <= 60 and re.match(r"^\d+(\.\d+)*\s+[A-Z]", s):
        return 2
    if s.lower() in (
        "abstract",
        "introduction",
        "related work",
        "method",
        "methods",
        "experiments",
        "experiment",
        "conclusion",
        "conclusions",
        "references",
        "reference",
        "acknowledgments",
    ):
        return 2
    return 0


def _rows_to_markdown(rows: list[list]) -> str:
    """把表格二维列表转为 Markdown 表格（首行表头），单元格转义竖线/换行。"""
    rows = [list(r) for r in (rows or []) if any(str(c).strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ").strip()  # noqa: E731
    lines = [
        "| " + " | ".join(cell(c) for c in rows[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for r in rows[1:]:
        lines.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(lines)


def _table_looks_real(rows: list[list]) -> bool:
    """过滤误检的「伪表格」：无论有框线还是三线表启发式，都可能把正文/
    图内文字切成碎片行列。真表格的特征是：行列规整（矩形）、单元格短而
    成词（数值/方法名/指标名），几乎没有纯标点碎片单元格。"""
    rows = [list(r) for r in (rows or []) if any(str(c).strip() for c in r)]
    if len(rows) < 3:
        return False
    width = max(len(r) for r in rows)
    if width < 2:
        return False
    # 矩形性：碎片化检测会产生参差行列（各行列数不一致）
    ragged = sum(1 for r in rows if len(r) != width)
    if ragged > len(rows) * 0.2:
        return False
    cells = [str(c).strip() for r in rows for c in r if str(c).strip()]
    if len(cells) < 6:
        return False
    if max(len(c) for c in cells) > 40:
        return False
    # 纯标点碎片（"."、"-"、"，" 等）占比过高 = 数字被列切分打散
    punct = sum(1 for c in cells if re.fullmatch(r"[.\-–—×*·,，]+", c))
    if punct / len(cells) > 0.12:
        return False
    # 数字后缀句点（"910."、"81×."）= 小数点被当作列分隔的典型碎片
    frag_num = sum(1 for c in cells if c.endswith(".") and any(ch.isdigit() for ch in c[:-1]))
    if frag_num > max(1, len(cells) * 0.04):
        return False
    return (sum(len(c) for c in cells) / len(cells)) <= 16


def _build_full_translation(paper_id: str, target_lang: str) -> dict[str, Any] | None:
    """整篇翻译核心：抽取全文结构 -> 分块翻译 -> 组装原位排版 Markdown。

    正文/标题/图注/表格译为中文，公式/代码/参考文献保留原文，按页码锚点保留阅读顺序。
    """
    item = next((it for it in _load_registry() if it["paper_id"] == paper_id), None)
    if not item:
        return None
    path = os.path.join(UPLOAD_DIR, item["filename"])
    if not os.path.exists(path):
        return None
    try:
        chunks = _rag_mod.extract_chunks_from_pdf(path, min_len=4)
    except RuntimeError as e:
        return {"error": str(e)}
    if not chunks:
        return {"error": "未提取到文本块（可能为纯扫描件且 OCR 未就绪）"}

    # 结构抽取（与阅读器同源）
    formula_keys: set[tuple[Any, ...]] = set()
    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    toc: list[dict[str, Any]] = []
    total_pages = 0
    try:
        doc = fitz.open(path)
        total_pages = doc.page_count
        toc = [{"level": t[0], "title": t[1], "page": t[2]} for t in doc.get_toc(simple=False)]
        if not toc:
            toc = _build_heading_toc(doc)
        for f in _detect_formulas(doc):
            formula_keys.add((f["page"], tuple(f["bbox"])))
        tables, figures = _extract_tables_and_figures(doc)
        doc.close()
    except Exception:
        total_pages = max((c["page_num"] for c in chunks), default=0)

    toc_norm = [(t["title"].strip().lower().rstrip("."), t["level"], t["page"]) for t in toc]
    chunks_sorted = sorted(chunks, key=lambda c: (c["page_num"], c["bbox"][1]))

    title = _guess_title(chunks, item.get("title") or paper_id)

    # ---- 构建带类型的块序列 + 待翻译文本队列 ----
    blocks: list[dict[str, Any]] = []
    for c in chunks_sorted:
        text = c["text"].strip()
        if not text:
            continue
        page = c["page_num"]
        if c.get("is_ref"):
            blocks.append({"kind": "ref", "page": page, "text": text})
        elif (page, tuple(c["bbox"])) in formula_keys or c.get("is_formula"):
            blocks.append({"kind": "formula", "page": page, "text": text})
        else:
            lvl = _match_heading_level(text, toc_norm, page)
            if lvl:
                blocks.append({"kind": "heading", "page": page, "text": text, "level": lvl})
            else:
                blocks.append({"kind": "para", "page": page, "text": text})

    # 图表/表格按页挂接（原位：跟随所在页的正文块之后）
    # 表格分级导出：结构可信（检查通过）→ Markdown 表格（单元格译文）；
    # 结构可疑（列切分错位会产生乱码碎片）→ 直接嵌入区域截图（data URI，
    # 渲染永远正确）；三线表启发式检测 → 注释提示。
    page_tables: dict[int, list[list]] = {}
    page_table_imgs: dict[int, list[str]] = {}
    page_ruleless: dict[int, int] = {}
    page_figs: dict[int, list[str]] = {}
    for t in tables:
        p = int(t["page"])
        rows = t.get("rows") or []
        if t.get("ruleless"):
            page_ruleless[p] = page_ruleless.get(p, 0) + 1
            continue
        if _table_looks_real(rows):
            page_tables.setdefault(p, []).append(rows)
        elif t.get("image"):
            page_table_imgs.setdefault(p, []).append(t["image"])
    for f in figures:
        cap = (f.get("caption") or "").strip()
        # 过长「图注」多为正文段落被 _match_caption 误抓，跳过避免与正文重复
        if cap and len(cap) <= 120:
            page_figs.setdefault(int(f["page"]), []).append(cap)

    # ---- 收集待翻译文本（保持顺序，一次性批量翻译） ----
    queue: list[str] = [title]
    for b in blocks:
        if b["kind"] in ("heading", "para"):
            queue.append(b["text"])
    # 图注（按页序，与 page_figs 遍历顺序一致）
    fig_keys: list[tuple[int, int]] = []  # (page, 页内序号)
    for p in sorted(page_figs):
        for i, cap in enumerate(page_figs[p]):
            fig_keys.append((p, i))
            queue.append(cap)
    # 表格单元格（按页序/表序/行序/列序，与 page_tables 遍历顺序一致）
    for p in sorted(page_tables):
        for rows in page_tables[p]:
            for r in rows:
                for cell in r:
                    if str(cell).strip():
                        queue.append(str(cell).strip())

    translated = _translate_full_blocks(queue, target_lang)
    t_title = translated[0]

    # ---- 翻译结果按队列顺序切分回填：正文块 -> 图注 -> 表格单元格 ----
    # queue 顺序 = [title] + 正文块(页序) + 图注(页序) + 单元格(页/表/行/列序)，
    # 回填必须按同一顺序切分，否则图注/单元格会与正文错位。
    rest = translated[1:]
    n_body = sum(1 for b in blocks if b["kind"] in ("heading", "para"))
    t_body: list[str] = rest[:n_body]
    tail_iter = iter(rest[n_body:])
    t_fig_map: dict[tuple[int, int], str] = {}
    for key in fig_keys:
        t_fig_map[key] = next(tail_iter, "")
    t_tables: dict[int, list[list[str]]] = {}
    for p in sorted(page_tables):
        new_list: list[list[list[str]]] = []
        for rows in page_tables[p]:
            new_rows = [
                [next(tail_iter, str(c).strip()) if str(c).strip() else "" for c in r] for r in rows
            ]
            new_list.append(new_rows)
        t_tables[p] = new_list

    # ---- 组装 Markdown（按页分组；表格/图注每页只追加一次） ----
    md: list[str] = []
    md.append(f"# {t_title}")
    md.append("")
    md.append(f"<!-- 原题：{title} -->")
    md.append("<!-- 由 智研 Agent 整篇翻译，正文/标题/图注译为中文，公式/代码/参考文献保留原文 -->")
    md.append("")

    pages_order: list[int] = []
    blocks_by_page: dict[int, list[dict[str, Any]]] = {}
    for b in blocks:
        if b["page"] not in blocks_by_page:
            pages_order.append(b["page"])
            blocks_by_page[b["page"]] = []
        blocks_by_page[b["page"]].append(b)

    body_i = 0  # 正文块翻译游标（与 queue 中正文块顺序一致）
    for p in pages_order:
        md.append(f"<!-- ===== 第 {p} 页 ===== -->")
        for b in blocks_by_page[p]:
            if b["kind"] in ("heading", "para"):
                t = t_body[body_i] if body_i < len(t_body) else b["text"]
                body_i += 1
                md.append(f"{'#' * min(b.get('level', 2), 6)} {t}" if b["kind"] == "heading" else t)
            elif b["kind"] == "formula":
                md.append("```latex")
                md.append(b["text"])
                md.append("```")
            elif b["kind"] == "ref":
                md.append(f"> {b['text']}")
            md.append("")
        # 页内表格/图注原位追加（每页一次，不随段落块重复）
        for rows in t_tables.get(p, []):
            md.append(_rows_to_markdown(rows))
            md.append("")
        for img in page_table_imgs.get(p, []):
            md.append(f"![表（第 {p} 页）](data:image/png;base64,{img})")
            md.append("")
        if page_ruleless.get(p):
            md.append(
                f"<!-- 第 {p} 页含 {page_ruleless[p]} 个三线表（启发式检测，结构未还原；"
                f"截图预览见阅读器「表格图表」侧栏） -->"
            )
            md.append("")
        for i, cap in enumerate(page_figs.get(p, [])):
            label = t_fig_map.get((p, i), cap)
            md.append(f"> **图**（第 {p} 页）：{label}")
            md.append("")

    md_text = "\n".join(md).strip() + "\n"
    return {
        "markdown": md_text,
        "title": t_title,
        "source_title": title,
        "pages": total_pages,
        "blocks": len(blocks),
        "chars": len(md_text),
    }


class FullTranslateRequest(BaseModel):
    """整篇翻译请求：target_lang = zh（英->中）| en（中->英）。"""

    target_lang: str = "zh"


@app.post("/api/papers/{paper_id}/full-translate")
def full_translate_paper(paper_id: str, req: FullTranslateRequest):
    """整篇翻译（原位排版 + Markdown 导出）：模块三「整篇翻译」补全。

    正文/标题/图注译为中文，公式/代码/参考文献保留原文，按页码锚点保留阅读顺序；
    返回 Markdown 文本 + 元信息，前端可预览并下载 `.md`。
    """
    target_lang = req.target_lang if req.target_lang in ("zh", "en") else "zh"
    if getattr(_agent, "USE_MOCK_LLM", True):
        return JSONResponse(
            status_code=503, content={"error": "未配置 LLM（OPENAI_API_KEY），暂无法整篇翻译"}
        )
    res = _build_full_translation(paper_id, target_lang)
    if res is None:
        return JSONResponse(status_code=404, content={"error": "论文不存在或文件未入库"})
    if "error" in res:
        return JSONResponse(status_code=422, content={"error": res["error"]})
    return res


class TranslateRequest(BaseModel):
    """中英对照翻译：批量文本，返回与输入等长的译文列表。"""

    texts: list[str] = []
    target_lang: str = "zh"


@app.post("/api/tools/translate")
def translate_texts(req: TranslateRequest):
    """中英对照翻译：按块翻译，公式/代码/引用编号保留原文；返回等长译文列表。

    未配置 LLM 时返回占位提示（阅读器仍可打开，只是不显示译文）。
    """
    texts = [t for t in (req.texts or []) if str(t).strip()]
    if not texts:
        return {"translations": []}
    if getattr(_agent, "USE_MOCK_LLM", True):
        return {"translations": ["（未配置 LLM，无法翻译）"] * len(texts)}

    batch_size = 8
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prompt = (
            "把下面的英文段落逐段翻译成中文，用于学术论文的中英对照阅读。要求：\n"
            "1. 只输出译文，每段一行，不要编号、不要解释、不要输出原文；\n"
            "2. 公式、LaTeX、数字、变量名、代码、参考文献编号如 [12] 一律保留原文不译；\n"
            "3. 术语准确：attention=注意力、embedding=嵌入、token 保留英文，transformer 保留英文；\n"
            "4. 输出段落数量必须与输入完全一致。\n\n"
        )
        prompt += "\n".join(f"【段落{k + 1}】{t}" for k, t in enumerate(batch))
        try:
            resp = _agent.get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
            out.extend(_split_translation(resp, len(batch)))
        except Exception as e:  # noqa: BLE001 —— 单批失败不拖垮整页
            out.extend([f"（翻译失败：{str(e)[:60]}）"] * len(batch))
    return {"translations": out}


class TranslateEnRequest(BaseModel):
    """中译英学术翻译：中文摘要 / 全文 -> 符合学术规范的英文。"""

    text: str = ""
    target_lang: str = "en"


@app.post("/api/tools/translate-en")
def translate_zh_to_en_api(req: TranslateEnRequest):
    """模块六「学术翻译」：中文论文摘要 / 全文一键翻译为学术英文。

    复用 writing.translate_zh_to_en；保留公式 / LaTeX / 变量名 / 引用编号。
    """
    text = (req.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "请提供待翻译的中文内容"})
    if getattr(_agent, "USE_MOCK_LLM", True):
        return JSONResponse(
            status_code=503,
            content={
                "error": "未配置 LLM（OPENAI_API_KEY），暂无法翻译，请在 config.py 配置后重试"
            },
        )
    try:
        translated = writing.translate_zh_to_en(_agent.get_llm(), text)
        return {"translation": translated}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": f"翻译失败：{str(e)[:200]}"})


def _guess_title(chunks: list[dict[str, Any]], fallback: str) -> str:
    """从首页文本块猜测论文标题：取首页第一个较短的非声明类块
    （跳过版权声明/邮箱/页眉等，CVPR 标题通常独立成块）。"""
    for c in chunks:
        t = c["text"]
        if c["page_num"] == 1 and 10 <= len(t) < 200:
            low = t.lower()
            if "@" in t or "permission" in low or "copyright" in low or "attribution" in low:
                continue
            return t
    return fallback


def _summarize_chunks(chunks: list[dict[str, Any]], max_chars: int = 1200) -> str:
    """取前两页的非参考文献块拼成摘要文本，供语义查重嵌入。"""
    parts = []
    for c in chunks:
        if c["page_num"] <= 2 and not c.get("is_ref"):
            parts.append(c["text"])
        if sum(len(p) for p in parts) >= max_chars:
            break
    return " ".join(parts)[:max_chars]


def _bump_version(version: str) -> str:
    """版本号递增：v1 -> v2；无法解析时从 v2 起步。"""
    import re as _re

    m = _re.fullmatch(r"v(\d+)", str(version).strip())
    return f"v{int(m.group(1)) + 1}" if m else "v2"


def _dup_suggestion(existing_item: dict[str, Any] | None, chunks: list[dict[str, Any]]) -> str:
    """新旧版本智能建议：比较库中版本年份与新文件提取年份，
    能判断新旧时给出「建议保留/建议覆盖」，否则提示人工选择。"""
    if not existing_item:
        return "检测到高度重复，请选择保留库中版本或覆盖为新版本。"
    ver = existing_item.get("version", "v1")
    ey = str(existing_item.get("year") or "")
    # 从新文件前 10 块文本提取年份（取最大者作为疑似出版年）
    head = " ".join(c["text"] for c in chunks[:10])[:2000]
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", head)]
    ny = max(years) if years else None
    if ey.isdigit() and ny:
        if ny <= int(ey):
            return f"库中已有 {ey} 年版本（{ver}），新文件年份不更晚，【建议保留】库中版本。"
        return f"库中为 {ey} 年版本（{ver}），新文件疑似 {ny} 年更新版，【建议覆盖】为新版本。"
    return f"库中已有该论文（{ver}），无法确定新旧，请选择保留或覆盖。"


def _summarize_revision(old_text: str, new_text: str) -> str:
    """自动对比两版本并总结主要修订差异：行级统计兜底 + LLM 要点总结（可用时）。"""
    if not old_text:
        return ""
    diff = version_dedup.diff_versions(old_text, new_text)
    adds = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    dels = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    if adds == 0 and dels == 0:
        return "两版本开头内容完全一致，无明显修订。"
    stats = f"修订统计：新增 {adds} 行 / 删除 {dels} 行。"
    if getattr(_agent, "USE_MOCK_LLM", True):
        return stats
    try:
        summary = (
            _agent.get_llm()
            .invoke(
                [
                    HumanMessage(
                        content=(
                            "对比同一论文两个版本的开头内容，用 2-4 条中文要点总结主要修订差异"
                            "（方法 / 数据 / 结论的变化）。\n\n"
                            f"旧版本：\n{old_text[:1500]}\n\n新版本：\n{new_text[:1500]}"
                        )
                    )
                ]
            )
            .content.strip()
        )
        return f"{stats}\n{summary}"
    except Exception:
        return stats


def _replace_existing_version(
    dup: dict, chunks: list[dict[str, Any]], pid: str
) -> tuple[str, str, str]:
    """覆盖为新版本：生成演进 Diff + 自然语言修订总结、清理旧向量与登记、版本号递增。
    返回 (新版本号, 原始diff, 自然语言总结)。"""
    old = next(
        (it for it in _load_registry() if it["paper_id"] == dup["existing_paper_id"]),
        None,
    )
    old_text = _extract_head_text(old)
    new_text = " ".join(c["text"] for c in chunks[:10])[:3000]
    diff_text, diff_summary = "", ""
    if old_text:
        diff_text = version_dedup.diff_versions(
            old_text,
            new_text,
            from_name=dup["existing_paper_id"],
            to_name=pid,
        )[
            :6000
        ]  # 截断防止登记簿膨胀
        diff_summary = _summarize_revision(old_text, new_text)
    delete_paper(dup["existing_paper_id"])
    return _bump_version((old or {}).get("version", "v1")), diff_text, diff_summary


def _extract_head_text(registry_item: dict[str, Any] | None, max_chars: int = 3000) -> str:
    """重新解析旧版 PDF 的开头文本，用于生成版本演进 Diff。"""
    if not registry_item or _rag_mod is None:
        return ""
    path = os.path.join(UPLOAD_DIR, registry_item["filename"])
    if not os.path.exists(path):
        return ""
    chunks = _rag_mod.extract_chunks_from_pdf(path)
    return " ".join(c["text"] for c in chunks[:10])[:max_chars]


@app.post("/api/upload")
async def upload_paper(
    file: UploadFile = File(...),
    paper_id: str | None = Form(None),
    force: bool = Form(False),
):
    """
    上传单个 PDF 并解析入库：
      1. 保存文件到 uploads/；
      2. PyMuPDF 提取带页码坐标的文本块（扫描件无文本层会返回 422 提示走 OCR）；
      3. 【语义去重 + HITL】与库内论文算 Embedding 相似度，疑似旧版本返回 409，
         由前端弹窗人工确认（替换 / 跳过）；force=true 时执行替换：
         生成版本演进 Diff、清理旧向量与登记、版本号递增；
      4. 向量写入 Qdrant，元数据登记到 papers.json。
    """
    if _rag_mod is None or ingest_pdf_with_anchors is None:
        return JSONResponse(status_code=503, content={"error": "RAG 模块未就绪，无法入库"})
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "仅支持 PDF 文件"})

    pid = (paper_id or os.path.splitext(file.filename)[0]).strip()
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # 先解析计数：扫描件自动走 RapidOCR（未装 OCR 引擎时抛 RuntimeError，转 422 并给出安装命令）
    try:
        chunks = _rag_mod.extract_chunks_from_pdf(save_path)
    except RuntimeError as e:
        os.remove(save_path)
        return JSONResponse(status_code=422, content={"error": str(e)})
    if not chunks:
        os.remove(save_path)
        return JSONResponse(
            status_code=422,
            content={
                "error": "未提取到文本块：既无原生文本层，OCR 也没识别出内容，请确认文件不是空白/损坏的 PDF"
            },
        )

    # ---- 语义去重与 HITL 版本检查 ----
    dup = dedup_service.check_paper_duplicate(
        _rag_mod,
        pdf_title=_guess_title(chunks, pid),
        pdf_summary_text=_summarize_chunks(chunks),
    )
    same_paper_reupload = dup.get("existing_paper_id") == pid  # 同 paper_id 重传视为正常更新
    version, diff_text, diff_summary = "v1", "", ""

    if dup["is_duplicate"] and not same_paper_reupload:
        if not force:
            # 未确认：拦截入库，附新旧建议与修订差异总结，交前端人工决策（HITL）
            os.remove(save_path)
            old = next(
                (it for it in _load_registry() if it["paper_id"] == dup["existing_paper_id"]),
                None,
            )
            dup["suggestion"] = _dup_suggestion(old, chunks)
            old_text = _extract_head_text(old)
            dup["revision_summary"] = _summarize_revision(
                old_text, " ".join(c["text"] for c in chunks[:10])[:3000]
            )
            return JSONResponse(status_code=409, content=dup)
        # 用户确认"作为新版本替换"：演进 Diff + 修订总结 + 清旧数据 + 版本递增
        version, diff_text, diff_summary = _replace_existing_version(dup, chunks, pid)

    ingest_pdf_with_anchors(save_path, pid)

    # 登记簿：同名 paper_id 视为重传，更新记录
    _register_pdf(
        pid,
        file.filename,
        len(chunks),
        version=version,
        diff_text=diff_text,
        diff_summary=diff_summary,
    )
    _backfill_title(pid, chunks)  # 本地标题回填（无需网络）
    _enrich_metadata(save_path, pid)  # PDF 元数据识别：联网补全作者/期刊/年份/DOI
    return {
        "paper_id": pid,
        "filename": file.filename,
        "chunks": len(chunks),
        "version": version,
        "replaced": (
            dup.get("existing_paper_id")
            if (force and dup["is_duplicate"] and not same_paper_reupload)
            else None
        ),
    }


def _register_pdf(
    pid: str,
    filename: str,
    chunks: int,
    version: str = "v1",
    diff_text: str = "",
    diff_summary: str = "",
) -> int:
    """登记已入库论文（同名 paper_id 视为重传），返回当前库内论文总数。"""
    import datetime

    items = [it for it in _load_registry() if it["paper_id"] != pid]
    record: dict[str, Any] = {
        "paper_id": pid,
        "filename": filename,
        "chunks": chunks,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": version,
    }
    if diff_text:
        record["diff"] = diff_text  # 原始演进 Diff（技术细节，弹窗可折叠查看）
    if diff_summary:
        record["diff_summary"] = diff_summary  # 自然语言修订总结（主视图）
    items.append(record)
    _save_registry(items)
    return len(items)


def _backfill_title(pid: str, chunks: list[dict[str, Any]]):
    """本地标题回填（无需网络）：用首页首个短块作为论文标题写入登记簿，
    保证右击知识库装入提问框时是可读标题而非文件名。"""
    items = _load_registry()
    for it in items:
        if it["paper_id"] == pid and not it.get("title"):
            it["title"] = _guess_title(chunks, pid)
            _save_registry(items)
            return


def _enrich_metadata(save_path: str, pid: str):
    """PDF 元数据识别（仿 Zotero 拖入识别）：从前两页文本提取 DOI，
    经 Crossref 补全 作者/期刊/年份/DOI/标题 写入登记簿；失败静默跳过。"""
    if _rag_mod is None:
        return
    try:
        chunks = _rag_mod.extract_chunks_from_pdf(save_path)
        head = " ".join(c["text"] for c in chunks if c["page_num"] <= 2)[:4000]
        m = re.search(r"10\.\d{4,9}/[^\s\"<>]+", head)
        if not m:
            return
        meta = scholar_sources.resolve_identifier(m.group(0).rstrip(".;,)"))
        items = _load_registry()
        for it in items:
            if it["paper_id"] == pid:
                it["authors"] = meta.get("authors", [])[:6]
                it["journal"] = meta.get("venue", "")
                it["year"] = meta.get("published", "")
                it["doi"] = meta.get("doi", "")
                if meta.get("title"):
                    it["title"] = meta["title"]
        _save_registry(items)
    except Exception:
        pass  # 网络不可达/无 DOI 不影响正常入库


def _ingest_downloaded_pdf(save_path: str, title: str | None) -> str:
    """把 arXiv 下载的 PDF 查重并解析入库（TOOL 闭环 / 一键导入共用），返回状态消息。"""
    if _rag_mod is None or ingest_pdf_with_anchors is None:
        return "RAG 模块未就绪，已下载的 PDF 未能入库。"
    try:
        chunks = _rag_mod.extract_chunks_from_pdf(save_path)
    except RuntimeError as e:
        return str(e)
    if not chunks:
        return "下载的 PDF 未提取到文本块（可能为扫描件且 OCR 无结果），未入库。"
    pid = (title or os.path.splitext(os.path.basename(save_path))[0])[:100]
    dup = dedup_service.check_paper_duplicate(
        _rag_mod, _guess_title(chunks, pid), _summarize_chunks(chunks)
    )
    if dup["is_duplicate"] and dup.get("existing_paper_id") != pid:
        return f"查重拦截：与库内「{dup['existing_paper_id']}」相似度 {dup['similarity']}，已跳过入库。"
    ingest_pdf_with_anchors(save_path, pid)
    total = _register_pdf(pid, os.path.basename(save_path), len(chunks))
    _backfill_title(pid, chunks)  # 本地标题回填（无需网络）
    return f"已自动入库：{pid}（{len(chunks)} 个文本块），知识库现有 {total} 篇。"


@app.post("/api/upload-batch")
async def upload_batch(files: list[UploadFile] = File(...), policy: str = Form("ask")):
    """
    批量 PDF 上传（基础 CRUD 之批量 Create）：
      policy = ask     -> 新论文直接入库，重复论文仅返回预警信息（不写入），
                          附新旧建议与修订差异总结，由前端统一决策；
      policy = skip    -> 重复论文跳过（保留库中版本）；
      policy = replace -> 重复论文覆盖为新版本（v1->v2，保留演进 Diff）。
    返回逐文件结果列表：status ∈ imported / duplicate / skipped / replaced / failed。
    """
    if _rag_mod is None or ingest_pdf_with_anchors is None:
        return JSONResponse(status_code=503, content={"error": "RAG 模块未就绪，无法入库"})

    results = []
    for file in files:
        name = file.filename or "unknown.pdf"
        if not name.lower().endswith(".pdf"):
            results.append({"filename": name, "status": "failed", "message": "非 PDF 文件"})
            continue
        pid = os.path.splitext(name)[0].strip()
        save_path = os.path.join(UPLOAD_DIR, name)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        try:
            chunks = _rag_mod.extract_chunks_from_pdf(save_path)
        except RuntimeError as e:
            os.remove(save_path)
            results.append({"filename": name, "status": "failed", "message": str(e)})
            continue
        if not chunks:
            os.remove(save_path)
            results.append(
                {
                    "filename": name,
                    "status": "failed",
                    "message": "未提取到文本块（无文本层且 OCR 无结果）",
                }
            )
            continue

        dup = dedup_service.check_paper_duplicate(
            _rag_mod, _guess_title(chunks, pid), _summarize_chunks(chunks)
        )
        same_pid = dup.get("existing_paper_id") == pid

        if dup["is_duplicate"] and not same_pid and policy != "replace":
            old = next(
                (it for it in _load_registry() if it["paper_id"] == dup["existing_paper_id"]),
                None,
            )
            dup["suggestion"] = _dup_suggestion(old, chunks)
            if policy == "ask":
                old_text = _extract_head_text(old)
                dup["revision_summary"] = _summarize_revision(
                    old_text, " ".join(c["text"] for c in chunks[:10])[:3000]
                )
                os.remove(save_path)
                results.append({"filename": name, "status": "duplicate", "dup": dup})
            else:  # skip：保留库中版本
                os.remove(save_path)
                results.append(
                    {
                        "filename": name,
                        "status": "skipped",
                        "message": f"与「{dup['existing_paper_id']}」重复（相似度 {dup['similarity']}），已按策略保留库中版本",
                        "dup": dup,
                    }
                )
            continue

        version, diff_text, diff_summary = "v1", "", ""
        replaced_from = None
        if dup["is_duplicate"] and not same_pid and policy == "replace":
            version, diff_text, diff_summary = _replace_existing_version(dup, chunks, pid)
            replaced_from = dup["existing_paper_id"]

        ingest_pdf_with_anchors(save_path, pid)
        _register_pdf(
            pid, name, len(chunks), version=version, diff_text=diff_text, diff_summary=diff_summary
        )
        _backfill_title(pid, chunks)
        _enrich_metadata(save_path, pid)
        status = "replaced" if replaced_from else "imported"
        results.append(
            {
                "filename": name,
                "status": status,
                "message": (
                    f"已作为 {version} 覆盖「{replaced_from}」（{len(chunks)} 块）"
                    if replaced_from
                    else f"已入库（{len(chunks)} 块）"
                ),
            }
        )
    return {"results": results}


# ============================================================
# CSV 数据上传：VISUAL 分支使用用户自己的数据绘图
# ============================================================
CSV_DIR = os.path.join(BASE_DIR, "uploads_csv")
os.makedirs(CSV_DIR, exist_ok=True)


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """上传 CSV 数据文件，返回服务端路径供后续绘图提问使用。"""
    if not file.filename.lower().endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "仅支持 CSV 文件"})
    save_path = os.path.join(CSV_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"csv_path": save_path, "filename": file.filename}


@app.post("/api/papers/{paper_id}/diff-summary")
def paper_diff_summary(paper_id: str):
    """惰性生成版本演进的自然语言总结：优先读缓存，否则基于已存原始 Diff
    用 LLM 总结（无 LLM 时退化为增删行统计），生成后写回登记簿缓存。"""
    items = _load_registry()
    item = next((it for it in items if it["paper_id"] == paper_id), None)
    if not item or not item.get("diff"):
        return JSONResponse(status_code=404, content={"error": "无版本演进记录"})
    if item.get("diff_summary"):
        return {"diff_summary": item["diff_summary"]}

    diff = item["diff"]
    adds = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    dels = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    summary = f"修订统计：新增 {adds} 行 / 删除 {dels} 行。"
    if not getattr(_agent, "USE_MOCK_LLM", True):
        try:
            text = (
                _agent.get_llm()
                .invoke(
                    [
                        HumanMessage(
                            content=(
                                "以下是同一论文两个版本的文本 Diff（- 为旧版本删除内容，+ 为新版本新增内容）。"
                                "请用 2-5 条中文要点，以自然语言总结两个版本的主要修订差异"
                                "（如结构 / 作者 / 方法 / 实验 / 结论的变化），不要复述原始文本：\n\n"
                                + diff[:4000]
                            )
                        )
                    ]
                )
                .content.strip()
            )
            summary = f"{summary}\n{text}"
        except Exception:
            pass  # LLM 失败时保留行级统计
    item["diff_summary"] = summary
    _save_registry(items)
    return {"diff_summary": summary}


@app.get("/api/papers/columns")
def paper_columns():
    """【预留】知识库自定义列元数据（前端「添加列」）。

    注意：必须注册在 /api/papers/{paper_id} 之前，否则会被动态路由抢先匹配。
    契约：{"columns": [{"key": str, "label": str, "type": "text" | "number" | "date"}]}
    """
    return _reserved('GET /api/papers/columns -> {"columns": [...]}')


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str):
    """CRUD-Read：查询单篇论文登记信息。"""
    item = next((it for it in _load_registry() if it["paper_id"] == paper_id), None)
    if not item:
        return JSONResponse(status_code=404, content={"error": "论文不存在"})
    return item


class PaperPatchRequest(BaseModel):
    """CRUD-Update：重命名 paper_id 或补全/修改标题。"""

    new_paper_id: str | None = None
    title: str | None = None


@app.patch("/api/papers/{paper_id}")
def patch_paper(paper_id: str, req: PaperPatchRequest):
    """CRUD-Update：修改论文登记信息；重命名时同步更新向量库 payload。"""
    items = _load_registry()
    item = next((it for it in items if it["paper_id"] == paper_id), None)
    if not item:
        return JSONResponse(status_code=404, content={"error": "论文不存在"})
    if req.new_paper_id and req.new_paper_id != paper_id:
        if any(it["paper_id"] == req.new_paper_id for it in items):
            return JSONResponse(status_code=409, content={"error": "目标名称已存在"})
        item["paper_id"] = req.new_paper_id
        if _rag_mod is not None:
            try:
                from qdrant_client.models import (  # pyright: ignore[reportMissingImports]
                    Filter,
                    FieldCondition,
                    MatchValue,
                )

                _rag_mod.qdrant.set_payload(
                    collection_name=_rag_mod.COLLECTION_NAME,
                    payload={"paper_id": req.new_paper_id},
                    points=Filter(
                        must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                    ),
                )
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"向量库重命名失败：{e}"})
    if req.title:
        item["title"] = req.title
    _save_registry(items)
    return {"ok": True, "paper": item}


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: str, remove_file: bool = False):
    """从登记簿与 Qdrant 向量库中同时删除指定论文；
    remove_file=true 时一并删除磁盘上的 PDF 原文件（默认保留，供演进 Diff 复查）。"""
    items = _load_registry()
    target = next((it for it in items if it["paper_id"] == paper_id), None)
    _save_registry([it for it in items if it["paper_id"] != paper_id])
    if remove_file and target:
        path = os.path.join(UPLOAD_DIR, target.get("filename", ""))
        if os.path.exists(path):
            os.remove(path)
    if _rag_mod is not None:
        try:
            from qdrant_client.models import (  # pyright: ignore[reportMissingImports]
                Filter,
                FieldCondition,
                MatchValue,
            )

            _rag_mod.qdrant.delete(
                collection_name=_rag_mod.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                ),
            )
        except Exception as e:
            return {"deleted": paper_id, "vector_cleanup": f"向量清理失败：{e}"}
    return {"deleted": paper_id}


# ============================================================
# TOOL 分支配套：arXiv 一键导入 + 最新论文定期抓取
# ============================================================
class ArxivImportRequest(BaseModel):
    """一键导入请求体：arXiv 论文卡片上的三要素。"""

    arxiv_id: str
    title: str
    pdf_url: str


@app.post("/api/tools/arxiv-import")
def arxiv_import(req: ArxivImportRequest):
    """一键导入闭环：下载 arXiv PDF -> 语义查重 -> 解析入库 -> 登记。"""
    if _rag_mod is None or ingest_pdf_with_anchors is None:
        return JSONResponse(status_code=503, content={"error": "RAG 模块未就绪，无法入库"})
    safe = re.sub(r'[\\/:*?"<>|]', "_", req.title)[:80] or req.arxiv_id
    save_path = os.path.join(UPLOAD_DIR, f"{safe}.pdf")
    try:
        arxiv_tool.download_pdf(req.pdf_url, save_path)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"下载失败（当前网络可能不可达 arXiv）：{e}"},
        )
    msg = _ingest_downloaded_pdf(save_path, req.title)
    return {"ok": msg.startswith("已自动入库"), "message": msg}


LATEST_FILE = os.path.join(BASE_DIR, "latest_papers.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "fetch_settings.json")


def _load_settings() -> dict[str, Any]:
    """读取抓取设置（关键词/来源/自定义数据源），缺省用 config.py 与库内偏好。

    custom_sources 为用户添加的自定义数据源列表：
      [{"id","name","url","api_key","doc_url"}]，url 中用 {query} 作关键词占位符；
    custom_source_ids 为「参与抓取」的数据源 id 列表（可多条，同时抓取）；
    custom_source_id 仅保留向后兼容（= custom_source_ids 的第一条）。
    """
    settings = {
        "keywords": getattr(config, "ARXIV_KEYWORDS", ""),
        "sources": ["arxiv"],
        "custom_api_url": "",
        "custom_sources": [],
        "custom_source_ids": [],
        "custom_source_id": "",
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    # 旧版本设置只有 custom_api_url：自动迁移成一条「自定义API」数据源
    if not settings.get("custom_sources") and settings.get("custom_api_url"):
        settings["custom_sources"] = [
            {
                "id": "src_legacy",
                "name": "自定义API",
                "url": settings["custom_api_url"],
                "api_key": "",
                "doc_url": "",
            }
        ]
        settings["custom_source_id"] = "src_legacy"
    if settings.get("custom_sources") and not settings.get("custom_source_id"):
        settings["custom_source_id"] = settings["custom_sources"][0].get("id", "")

    # 多源抓取：custom_source_ids 缺失（旧配置）时默认「全部启用」，
    # 与新版界面上每个数据源默认勾选的直觉一致；失效 id 直接剔除。
    valid_ids = [
        str(s.get("id") or "") for s in settings.get("custom_sources") or [] if isinstance(s, dict)
    ]
    ids = [str(i) for i in (settings.get("custom_source_ids") or []) if str(i or "").strip()]
    ids = [i for i in ids if i in valid_ids]
    if not ids and valid_ids:
        ids = valid_ids
    settings["custom_source_ids"] = ids
    if ids:
        settings["custom_source_id"] = ids[0]  # 兼容字段跟随多选结果，避免返回前后矛盾
    return settings


def _save_settings(settings: dict[str, Any]):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


class FetchSettingsRequest(BaseModel):
    """抓取设置：关键词、来源列表（arxiv/pubmed/custom）、自定义数据源。"""

    keywords: str = ""
    sources: list[str] = ["arxiv"]
    # 兼容旧字段（单一自定义 URL），新前端改用 custom_sources + custom_source_ids
    custom_api_url: str = ""
    custom_sources: list[dict[str, Any]] = []
    # None = 没传（旧前端 / 兼容），按「全部启用」处理；[] = 显式一个都不选
    custom_source_ids: list[str] | None = None
    custom_source_id: str = ""


@app.get("/api/tools/settings")
def get_settings():
    """返回当前抓取设置与生效关键词，供前端面板初始化。

    api_version 用于前端判断「后端进程是不是旧代码」：
    页面每次都从磁盘读最新 index.html，但 uvicorn 不重启就不会加载新 app.py，
    两者版本不一致时前端会直接提示重启，避免出现「界面是新的、逻辑是旧的」的怪现象。
    """
    return {
        "settings": _load_settings(),
        "effective_keywords": _fetch_keywords(),
        "api_version": API_VERSION,
    }


@app.post("/api/tools/settings")
def save_settings(req: FetchSettingsRequest):
    """保存抓取设置（定时任务与手动刷新共用同一设置）。

    自定义数据源：前端「添加数据源」弹窗维护 custom_sources 列表，
    custom_source_ids 指定本次参与抓取的若干条（可多选，会同时抓取并合并结果）。
    """
    sources = req.sources or ["arxiv"]

    # 清洗数据源：丢掉没有 URL 的空条目，补齐 id / name / 时间窗
    cleaned: list[dict[str, Any]] = []
    for i, s in enumerate(req.custom_sources or []):
        url = str(s.get("url", "")).strip()
        if not url:
            continue
        try:
            days = int(s.get("days") or 0)
        except (TypeError, ValueError):
            days = 0
        cleaned.append(
            {
                "id": str(s.get("id") or f"src_{i}_{abs(hash(url)) % 100000}"),
                "name": str(s.get("name") or url[:40]).strip(),
                "url": url,
                "api_key": str(s.get("api_key", "")).strip(),
                "doc_url": str(s.get("doc_url", "")).strip(),
                # days = 只要最近 N 天的结果，0 表示不限；决定 URL 里 {from} 的取值
                "days": max(0, min(3650, days)),
            }
        )

    # 多选：只接受仍存在的 id，顺序按请求顺序；None = 旧前端没传 -> 全部启用
    enabled: list[str] | None
    if req.custom_source_ids is None:
        enabled = [c["id"] for c in cleaned]
    else:
        enabled = [str(i).strip() for i in req.custom_source_ids if str(i or "").strip()]
        enabled = [i for i in enabled if any(c["id"] == i for c in cleaned)]

    if "custom" in sources:
        if not cleaned and not req.custom_api_url.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "已勾选「自定义API」，请先点「＋ 添加」创建一个数据源"},
            )
        if cleaned and not enabled:
            return JSONResponse(
                status_code=400,
                content={"error": "已勾选「自定义API」，请在下拉列表里至少勾选一个数据源"},
            )

    selected = enabled[0] if enabled else ""
    _save_settings(
        {
            "keywords": req.keywords.strip(),
            "sources": sources,
            "custom_api_url": req.custom_api_url.strip(),
            "custom_sources": cleaned,
            "custom_source_ids": enabled or [],
            "custom_source_id": selected,
        }
    )
    return {
        "ok": True,
        "settings": _load_settings(),
        "effective_keywords": _fetch_keywords(),
    }


class SourceTestRequest(BaseModel):
    """数据源连通性测试：直接拿弹窗里还没保存的表单内容跑一次检索。"""

    url: str = ""
    api_key: str = ""
    days: int = 0
    query: str = ""


@app.post("/api/tools/test-source")
def test_custom_source(req: SourceTestRequest):
    """用当前表单内容实际检索一次，返回前 3 条结果与 PDF 命中情况。

    用于「添加数据源」弹窗的「测试检索」按钮：先验证 URL / Key / 占位符是否正确，
    再保存，避免存进一条抓不到数据的坏数据源。
    """
    url = (req.url or "").strip()
    if not url:
        return JSONResponse(status_code=400, content={"error": "请先填写 API 地址"})
    try:
        # 关键词兜底也放在 try 内：否则 _fetch_keywords 出问题会变成 500，前端无从得知原因
        query = (req.query or "").strip() or _fetch_keywords()
        papers = scholar_sources.search_custom(
            url,
            query,
            max_results=3,
            api_key=(req.api_key or "").strip(),
            days=max(0, min(3650, req.days or 0)),
        )
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"检索失败：{str(e)[:220]}"})
    return {
        "ok": True,
        "query": query,
        "count": len(papers),
        "pdf_count": sum(1 for p in papers if p.get("pdf_url")),
        "papers": [
            {k: p.get(k, "") for k in ("title", "published", "pdf_url", "doi", "venue")}
            for p in papers
        ],
    }


def _fetch_keywords() -> str:
    """生效关键词：用户设置 > config.py > 库内偏好提炼。"""
    kw = _load_settings().get("keywords", "")
    if kw:
        return kw
    lib = _lib_keywords(4)
    return " ".join(lib) if lib else "skeleton action recognition"


_TITLE_STOP = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "with",
    "from",
    "by",
    "via",
    "using",
    "based",
    "towards",
    "toward",
    "into",
    "is",
    "are",
    "be",
    "as",
    "we",
    "its",
    "their",
    "paper",
    "study",
    "approach",
    "method",
    "methods",
    "analysis",
    "new",
    "novel",
    "review",
    "survey",
    "strong",
    "weak",
    "large",
    "small",
    "all",
    "you",
    "need",
    "can",
    "may",
    "must",
    "should",
    "will",
    "not",
    "more",
    "most",
    "less",
    "than",
    "such",
    "also",
    "both",
    "each",
    "other",
    "same",
    "only",
    "very",
}


def _keywords_from_title(title: str) -> str:
    """从论文标题提取「研究方向」检索词。

    整句标题直接去检索只能命中原文本身，所以取最长的 3 个实词当作术语
    （长词多为专业名词：multicomplexes / hyperkähler / manifolds），再按原顺序拼回。
    全大写单词降为小写，避免检索式看起来像喊话；带连字符的缩写（ST-GCN）保持原样。
    """
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-]{3,}", title or "")
    keep = [t for t in tokens if t.lower() not in _TITLE_STOP and not t.isdigit()]
    if not keep:
        return (title or "").strip()[:80]
    chosen = set(sorted(keep, key=len, reverse=True)[:3])
    out: list[str] = []
    seen: set[str] = set()
    for t in keep:
        if t not in chosen:
            continue
        word = t.lower() if (t.isupper() and t.isalpha() and len(t) > 3) else t
        if word.lower() not in seen:
            seen.add(word.lower())
            out.append(word)
    return " ".join(out)


def _research_query_from_paper(title: str) -> str:
    """把某篇论文的标题转成「同一研究方向」的检索词。

    优先让 LLM 提炼关键词（更接近研究方向而不是标题原句）；
    离线 Mock / LLM 失败 / 输出含中文时，退回标题实词过滤，保证任何环境都能用。
    """
    title = (title or "").strip()
    if not title:
        return ""
    if not getattr(_agent, "USE_MOCK_LLM", True):
        prompt = (
            "下面是一篇论文的标题。请提炼 4~8 个英文关键词，用于检索「同一研究方向」的最新论文。"
            "只输出关键词本身，用空格分隔，不要解释，不要输出标题原句：\n" + title
        )
        try:
            kw = _agent.get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
            kw = re.sub(r"\s+", " ", kw).strip(" \"'`")
            # 含中文说明没按要求输出，直接退回词法方案
            if kw and len(kw) <= 200 and not re.search(r"[一-鿿]", kw):
                return kw
        except Exception:
            pass
    return _keywords_from_title(title)


def fetch_latest_papers(paper_id: str = "", keywords_override: str = "") -> dict[str, Any]:
    """按用户设置从多来源抓取最新论文并缓存（定期任务与手动刷新共用）。

    paper_id 非空时切换为「按该论文的相关方向检索」：
    从登记簿取标题 -> 提炼研究方向关键词 -> 用它代替全局关键词检索，
    返回结果里带上 mode / ref_paper 供前端标注来源。
    keywords_override 为前端面板里当前输入的关键词，仅用于本次抓取，不写回配置。
    """
    import datetime

    settings = _load_settings()
    # 关键词优先用前端本次提交的值：用户在输入框改了词点「立即刷新」就该立刻生效，
    # 不必先点「保存设置」（保存只影响定时抓取用的常驻关键词）。
    mode, ref_title = "keyword", ""
    keywords = (keywords_override or "").strip() or _fetch_keywords()

    if paper_id:
        meta = next((p for p in _load_registry() if p.get("paper_id") == paper_id), None)
        if meta:
            title = (meta.get("title") or "").strip()
            # 登记簿里的 title 是回填时截断过的，长标题要退回 paper_id（通常是完整标题）
            pid_txt = re.sub(r"\.pdf$", "", (meta.get("paper_id") or "").strip(), flags=re.I)
            pid_txt = re.sub(r"v\d+$", "", pid_txt).strip() or pid_txt
            ref_title = pid_txt if len(pid_txt) > len(title) else (title or pid_txt)
            derived = _research_query_from_paper(ref_title)
            if derived:
                mode, keywords = "paper", derived

    result: dict[str, Any] = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": mode,
        "ref_paper": ref_title,
        "ref_paper_id": paper_id if mode == "paper" else "",
        "keywords": keywords,
        "sources": settings.get("sources", ["arxiv"]),
        "papers": [],
        "error": None,
    }

    def _run(kw: str):
        return scholar_sources.search_sources(
            result["sources"],
            kw,
            max_results=8,
            custom_api_url=settings.get("custom_api_url", ""),
            custom_sources=settings.get("custom_sources", []),
            custom_source_id=settings.get("custom_source_id", ""),
            custom_source_ids=settings.get("custom_source_ids") or None,
        )

    papers, errors = _run(result["keywords"])

    # 「按论文方向」的检索词来自标题，往往偏窄（可能只命中原文本身）：
    # 命中不足 3 条时放宽到前两个词再试一次；再放宽到单词会引入同形异义词（科研语境噪声大），故不做。
    if mode == "paper" and len(papers) < 3:
        words = result["keywords"].split()
        if len(words) > 2:
            broader = " ".join(words[:2])
            more, errors2 = _run(broader)
            if len(more) > len(papers):
                papers, errors = more, errors2
                result["keywords"] = broader

    # 参考论文本身不算「相关方向的新论文」，从结果里剔掉
    if mode == "paper" and ref_title:

        def _norm_title(s: str) -> str:
            return re.sub(r"\W+", "", (s or "").lower())

        def _same_work(x: str, y: str) -> bool:
            """标题比较：登记簿 title 可能被截断，允许前缀包含。"""
            if not x or not y:
                return False
            if x == y:
                return True
            short, long_ = (x, y) if len(x) <= len(y) else (y, x)
            return len(short) >= 18 and long_.startswith(short)

        targets = [t for t in {_norm_title(ref_title), _norm_title(paper_id)} if t]
        papers = [
            p
            for p in papers
            if not any(_same_work(_norm_title(p.get("title")), t) for t in targets)
        ]

    result["papers"] = papers
    result["error"] = errors or None
    with open(LATEST_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


class IdentifierRequest(BaseModel):
    """标识符解析请求：DOI / arXiv ID / PubMed ID 原文。"""

    identifier: str


@app.post("/api/tools/resolve")
def resolve_identifier(req: IdentifierRequest):
    """标识符导入第一步：识别 DOI/arXiv ID/PubMed ID 并联网补全元数据。"""
    try:
        meta = scholar_sources.resolve_identifier(req.identifier)
        return {"ok": True, "paper": meta}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


class RemotePdfImportRequest(BaseModel):
    """远程 PDF 导入请求：标题 + 开放 PDF 链接（来源不限）。"""

    title: str
    pdf_url: str


@app.post("/api/tools/import-remote-pdf")
def import_remote_pdf(req: RemotePdfImportRequest):
    """通用远程 PDF 导入闭环：下载 -> 查重 -> 解析入库（标识符/批量导入共用）。"""
    if _rag_mod is None or ingest_pdf_with_anchors is None:
        return JSONResponse(status_code=503, content={"error": "RAG 模块未就绪，无法入库"})
    safe = re.sub(r'[\\/:*?"<>|]', "_", req.title)[:80]
    save_path = os.path.join(UPLOAD_DIR, f"{safe}.pdf")
    try:
        arxiv_tool.download_pdf(req.pdf_url, save_path)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"下载失败（网络或链接不可用）：{e}"},
        )
    msg = _ingest_downloaded_pdf(save_path, req.title)
    return {"ok": msg.startswith("已自动入库"), "message": msg}


@app.post("/api/tools/parse-bibfile")
async def parse_bibfile(file: UploadFile = File(...)):
    """批量导入第一步：解析 RIS / BibTeX 文件，返回条目列表（含可导入标记）。"""
    if not file.filename.lower().endswith((".ris", ".bib", ".txt")):
        return JSONResponse(status_code=400, content={"error": "仅支持 .ris / .bib / .txt 文件"})
    text = (await file.read()).decode("utf-8", "ignore")
    entries = bibparse.parse_bib_file(file.filename, text)
    # 引用含金量：文件仅含元数据、无正文，此时使用强度按中性计分（只判元数据与可核验性）
    quality = citation_quality.assess(entries)
    for i, e in enumerate(entries):
        e["importable"] = bool(e.get("pdf_url"))
        q = quality["items"][i] if i < len(quality["items"]) else None
        e["quality"] = (
            {
                "score": q["score"],
                "tier": q["tier"],
                "tier_label": q["tier_label"],
                "keep": q["keep"],
                "reasons": q["reasons"][:4],
                "flags": q["flags"],
            }
            if q
            else None
        )
    return {
        "count": len(entries),
        "entries": entries,
        "quality_summary": quality["summary"],
    }


@app.get("/api/tools/latest")
def get_latest():
    """返回最近一次定期抓取结果（未抓取过时返回提示）。"""
    if os.path.exists(LATEST_FILE):
        with open(LATEST_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "time": None,
        "mode": "keyword",
        "ref_paper": "",
        "keywords": _fetch_keywords(),
        "papers": [],
        "error": "尚未执行过抓取",
    }


@app.post("/api/tools/latest/refresh")
def refresh_latest(payload: dict[str, Any] | None = None):
    """手动立即刷新最新论文列表。

    body 可选：
      {"paper_id": "..."}  按该论文的研究方向检索最新论文（优先）；
      {"keywords": "..."}  用面板里当前输入的关键词检索（不写回配置）；
      都不传时用已保存的关键词，与定时抓取行为一致。
    """
    paper_id, keywords = "", ""
    if isinstance(payload, dict):
        paper_id = str(payload.get("paper_id") or "").strip()
        keywords = str(payload.get("keywords") or "").strip()
    return fetch_latest_papers(paper_id=paper_id, keywords_override=keywords)


# ============================================================
# 新前端「预留接口」
# ------------------------------------------------------------
# 前端（项目 / 搜索 / 科研工具页）已完成交互与调用，
# 后端按下面的入参-出参契约补齐实现后，前端无需改动即可生效。
# 未实现时统一返回 501 + {"reserved": true, "message": ...}，
# 前端据此自动降级：项目 -> localStorage；科研工具 -> 提示「待接入」。
# ============================================================
_RESERVED_MESSAGE = "该能力接口已预留，后端实现后前端无需改动即可生效"


def _reserved(contract: str) -> JSONResponse:
    """统一的「已预留未实现」响应（HTTP 501）。"""
    return JSONResponse(
        status_code=501,
        content={"reserved": True, "message": f"{_RESERVED_MESSAGE}｜契约：{contract}"},
    )


@app.get("/api/projects")
def list_projects():
    """【预留】科研项目列表（云端同步）。

    契约：{"projects": [{"id": str, "name": str, "paper_ids": [str],
                        "created": int, "updated": int, "archived": bool}]}
    """
    return _reserved('GET /api/projects -> {"projects": [...]}')


@app.post("/api/projects")
def save_projects(payload: dict[str, Any] | None = None):
    """【预留】整表保存项目（前端当前以 localStorage 兜底）。

    契约：入参 {"projects": [...]}，返回 {"ok": true}
    """
    return _reserved('POST /api/projects {"projects": [...]} -> {"ok": true}')


@app.post("/api/search")
def semantic_search(payload: dict[str, Any] | None = None):
    """【预留】知识库语义检索（前端搜索页当前用本地标题过滤兜底）。

    契约：入参 {"query": str, "top_k": int}
         返回 {"hits": [{"paper_id": str, "page_num": int, "text": str, "score": float}]}
    """
    return _reserved('POST /api/search {"query","top_k"} -> {"hits": [...]}')


class CitationQualityRequest(BaseModel):
    """引用含金量分析请求。

    reference_text / bib_text / entries 三选一（优先级：entries > reference_text > bib_text）：
      - entries：已解析条目（RIS/BibTeX 解析结果或检索结果），字段同 bibparse 输出；
      - reference_text：用户粘贴的参考文献列表（支持 [1] 编号 / BibTeX / RIS）；
      - bib_text：旧契约字段名，等同 reference_text（兼容早期前端）。
    body_text 可选：论文正文，用于统计每条文献被正文真实引用的次数；
    threshold 可选：保留阈值（默认 50），低于它的条目 keep=false（建议剔除）。
    """

    entries: list[dict[str, Any]] | None = None
    reference_text: str | None = None
    bib_text: str | None = None
    body_text: str | None = None
    threshold: int = 50


@app.post("/api/tools/truecite")
def truecite(req: CitationQualityRequest):
    """TrueCite：引用含金量评估与「应付式引用」过滤（赛题模块四）。

    打分维度：元数据完整度 25 + 可核验性 20 + 发表形态 15 + 正文使用强度 25 + 时效性 10，
    另有重复条目 / 占位题名 / 年份异常等扣分项，逐项理由随结果返回（可解释、可离线）。
    """
    reference_text = (req.reference_text or req.bib_text or "").strip()
    if req.entries:
        entries = req.entries
    elif reference_text:
        entries = citation_quality.parse_reference_text(reference_text)
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "请提供 entries（已解析条目）或 reference_text（参考文献原文）"},
        )
    if not entries:
        return JSONResponse(
            status_code=400, content={"error": "未能从输入中解析出任何参考文献条目"}
        )

    result = citation_quality.assess(
        entries,
        body_text=(req.body_text or "").strip(),
        threshold=max(0, min(100, int(req.threshold))),
    )
    return {"ok": True, **result}


@app.post("/api/tools/figure-to-ppt")
def figure_to_ppt(payload: dict[str, Any] | None = None):
    """【预留】论文插图转 PPT：从 PDF 提取 Figure/Table 选区并生成可编辑版式。

    契约：入参 {"paper_id": str, "page_num": int | None, "bbox": [x0, y0, x1, y1] | None}
         返回 {"ok": bool, "task_id": str, "download_url": str}
    """
    return _reserved('POST /api/tools/figure-to-ppt {"paper_id","page_num","bbox"}')


@app.post("/api/tools/figure-to-excel")
def figure_to_excel(payload: dict[str, Any] | None = None):
    """【预留】论文图表转 Excel：识别 PDF 表格选区内容并转为可编辑表格。

    契约：入参 {"paper_id": str, "page_num": int | None, "bbox": [x0, y0, x1, y1] | None}
         返回 {"ok": bool, "task_id": str, "download_url": str, "rows": int}
    """
    return _reserved('POST /api/tools/figure-to-excel {"paper_id","page_num","bbox"}')


@app.post("/api/papers/export")
def export_papers(payload: dict[str, Any] | None = None):
    """【预留】知识库「传输 / 导出」：按选中论文导出题录（BibTeX / RIS / CSV）。

    契约：入参 {"paper_ids": [str], "format": "bibtex" | "ris" | "csv"}
         返回 {"ok": bool, "download_url": str, "count": int}
    """
    return _reserved('POST /api/papers/export {"paper_ids","format"}')


@app.get("/api/trash")
def list_trash():
    """【预留】回收站列表（软删除的论文 / 项目，默认保留 30 天）。

    契约：{"items": [{"type": "paper" | "project", "id": str, "deleted_at": int}]}
    """
    return _reserved('GET /api/trash -> {"items": [...]}')


def _latest_scheduler():
    """后台线程：按 config.FETCH_INTERVAL_HOURS 定期抓取最新领域论文。
    config.FETCH_ENABLED=False 时跳过抓取（断网/演示场景避免反复失败刷日志），
    手动点击「最新论文」刷新不受此开关限制。"""
    import threading
    import time

    def loop():
        warned = False
        while True:
            interval = max(1, getattr(config, "FETCH_INTERVAL_HOURS", 6))
            if getattr(config, "FETCH_ENABLED", True):
                try:
                    fetch_latest_papers()
                except Exception as e:
                    print(f"[定期抓取] 失败：{str(e)[:150]}")
            elif not warned:
                print("[定期抓取] 已通过 config.FETCH_ENABLED 关闭，仅支持手动刷新")
                warned = True
            time.sleep(interval * 3600)

    threading.Thread(target=loop, daemon=True).start()


_latest_scheduler()


# ============================================================
# 一键启动入口：python app.py
# 等价于在本目录执行 uvicorn app:app --host 127.0.0.1 --port 8000
# 说明：此处不启用 reload —— reload 会二次导入模块，导致定时抓取线程被重复启动
# ============================================================
if __name__ == "__main__":
    import uvicorn

    print("[智研 Agent] 服务启动中，稍后访问 http://127.0.0.1:8000 （按 Ctrl+C 停止）")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
