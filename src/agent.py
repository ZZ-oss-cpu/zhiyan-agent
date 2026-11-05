# agent.py
# ============================================================
# 智研 Agent —— 基于 LangGraph 的多 Agent 路由工作流（方案一核心中枢）
#
# 功能：识别用户意图并分发到对应子 Agent 节点：
#   - RAG    : 论文细节问答，附带 page_num / bbox 溯源锚点（抗幻觉）
#   - GRAPH  : 引用图谱挖掘（PageRank + 介数中心性挖基石论文）
#   - VISUAL : CSV 数据智能绘图沙箱（Matplotlib -> Base64）
#
# 对应 PDF 方案一「核心代码 2：基于 LangGraph 的路由与任务规划 Agent」，
# 并补充了 Human-in-the-Loop 打断（图谱分析前需人工确认）与 __main__ 演示入口。
# ============================================================
# 延迟注解求值：允许在低版本 Python 目标下使用 list[...] 等新式泛型注解
from __future__ import annotations

import os
import re
import importlib.util
from typing import Any, TypedDict, Annotated, Sequence, Optional
import operator

import arxiv_tool  # 外部学术 API 工具（arXiv 检索/下载 + GitHub 复现性）

# pyright 定向抑制：IDE 分析环境解释器未装依赖时误报，运行时依赖真实存在
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)  # pyright: ignore[reportMissingImports]
from langgraph.graph import StateGraph, END  # pyright: ignore[reportMissingImports]

# pyright 定向抑制：IDE 分析环境的解释器若未安装 langgraph 会误报，
# 运行时依赖真实存在（langgraph>=0.2），不影响执行
from langgraph.checkpoint.memory import MemorySaver  # pyright: ignore[reportMissingImports]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # 仓库根目录（src/ 的上级）

# 加载 config.py 中的 API Key 配置（注入环境变量，优先级低于系统环境变量）
import config

config.apply()

# ------------------------------------------------------------
# LLM 配置说明：
#   - Key 来源优先级：系统环境变量 > config.py > 离线 Mock；
#   - 默认走 OpenAI 官方 API；使用 DeepSeek / 通义等国产兼容接口时，
#     在 config.py 中同时填写 OPENAI_BASE_URL 即可；
#   - 若未配置 OPENAI_API_KEY，则自动降级为离线 _MockLLM，
#     保证演示流程不联网也能跑通（真实效果请配置 Key 后重跑）。
# ------------------------------------------------------------
USE_MOCK_LLM = not os.environ.get("OPENAI_API_KEY")


class _MockLLM:
    """离线占位 LLM：基于关键词的意图路由 + 模板回答，仅用于无 API Key 时演示流程。"""

    def invoke(self, messages):
        text = messages[-1].content if messages else ""
        # 命中路由提示词 -> 用关键词模拟意图分类
        if "分类结果:" in text:
            # 只提取"用户输入"段做匹配，避免命中提示词选项说明里的关键词
            user_part = text.split("用户输入:")[-1].split("分类结果:")[0]
            # 多目标命中 -> PLAN 复合任务
            groups = [
                any(k in user_part for k in ("核心方法", "异同", "对比", "差异")),
                any(k in user_part for k in ("引用", "基石", "图谱")),
                any(k in user_part for k in ("综述",)),
                any(k in user_part for k in ("Future Work", "研究空白", "展望")),
            ]
            if sum(groups) >= 2:
                return AIMessage(content="PLAN")
            if any(k in user_part for k in ("引用", "基石", "图谱")):
                return AIMessage(content="GRAPH")
            # DIAGRAM 优先于 VISUAL：拓扑图脚本类问题更具体，避免被「画」误判为 CSV 绘图
            if any(
                k in user_part
                for k in (
                    "拓扑图",
                    "TikZ",
                    "tikz",
                    "mermaid",
                    "graphviz",
                    "架构图",
                    "流程图",
                    "脚本",
                )
            ):
                return AIMessage(content="DIAGRAM")
            if any(
                k in user_part
                for k in (
                    "框架",
                    "Abstract",
                    "Related Work",
                    "论文框架",
                    "idea",
                    "Idea",
                    "写作",
                    "References",
                )
            ):
                return AIMessage(content="WRITE")
            if any(k in user_part for k in ("画", "绘图", "图表", "CSV", "csv")):
                return AIMessage(content="VISUAL")
            if any(k in user_part for k in ("arXiv", "arxiv", "搜索", "下载", "最新论文", "推荐")):
                return AIMessage(content="TOOL")
            return AIMessage(content="RAG")
        # 命中 RAG 问答提示词 -> 模板回答
        if "仅基于以下论文片段" in text:
            return AIMessage(
                content="（离线 Mock 回答）已收到溯源片段，配置 OPENAI_API_KEY 后此处输出真实回答。"
            )
        return AIMessage(content="（离线 Mock）未识别的提示词。")


def get_llm():
    """按需返回真实 ChatOpenAI 或离线 _MockLLM（延迟构造，避免无 Key 时导入即报错）。"""
    if USE_MOCK_LLM:
        return _MockLLM()
    # 延迟 import：只有在有 Key 时才实例化真实 LLM（可选依赖，未安装时走 Mock）
    from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]

    # 模型名从 config.py 读取（如 deepseek-flash），未配置时回退 gpt-4o-mini
    # streaming=True：让节点内的 invoke() 也走流式 API 并 emit token 回调，
    # 供 front_window/app.py 的 astream_events 捕获实现 Token 级流式回答；
    # invoke() 结果仍会聚合为完整 AIMessage，翻译/改写等其它接口行为不变。
    return ChatOpenAI(model=config.LLM_MODEL or "gpt-4o-mini", temperature=0, streaming=True)


# ============================================================
# 子功能模块加载
# 各功能模块与本文件同位于 src/ 下，这里用 importlib 按路径加载。
# 若模块初始化失败（如无 API Key 导致 Embedding 构造失败、缺依赖等），
# 自动降级为安全兜底函数，保证路由主流程可用。
# ============================================================
def _load_module(mod_name: str, filename: str):
    """按文件路径动态加载 src/ 下的功能模块。"""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(BASE_DIR, filename))
    # 防御：路径非法时 spec 或 loader 可能为 None，提前抛出可读错误
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {mod_name}: {filename}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    # RAG 溯源模块：PyMuPDF 提取带页码坐标文本块 + Qdrant 向量检索
    _rag = _load_module("rag_ingestion", "source_tracing.py")
    retrieve_with_anchors = _rag.retrieve_with_anchors
    fetch_paper_head_chunks = _rag.fetch_paper_head_chunks
except Exception as e:
    print(f"[警告] RAG 溯源模块加载失败（{e}），RAG 分支将返回空检索结果。")

    def retrieve_with_anchors(query, paper_id=None, top_k=3):
        """兜底检索：返回空列表，rag_node 会据此提示未检索到内容。"""
        return []

    def fetch_paper_head_chunks(paper_id, max_chunks=2):
        """兜底：无向量库时返回空列表。"""
        return []


# 全文级提问识别：此类问题需要覆盖摘要/方法等更多章节，而非仅 top-5 片段
FULLTEXT_RE = re.compile(
    r"核心方法|主要内容|主要贡献|总结|综述|概括|整体|框架|方法是什么|讲一下|介绍"
)


try:
    # 图谱挖掘模块：NetworkX PageRank + 介数中心性
    _graph = _load_module("graph_mining", "graph_generation.py")
    build_and_analyze_citation_graph = _graph.build_and_analyze_citation_graph
except Exception as e:
    print(f"[警告] 图谱模块加载失败（{e}），GRAPH 分支不可用。")

    def build_and_analyze_citation_graph(data, top_k=2):
        return {"cornerstones": [], "graph_data": {"nodes": [], "edges": []}}


try:
    # 可视化模块：CSV -> Matplotlib -> Base64
    _viz = _load_module("python_executor", "visualization.py")
    execute_data_visualization = _viz.execute_data_visualization
    execute_data_visualization_v2 = _viz.execute_data_visualization_v2
except Exception as e:
    print(f"[警告] 可视化模块加载失败（{e}），VISUAL 分支不可用。")

    def execute_data_visualization(csv_path, user_instruction):
        raise RuntimeError("可视化模块未就绪")

    def execute_data_visualization_v2(csv_path, user_instruction, caption_fn=None):
        raise RuntimeError("可视化模块未就绪")


# 学术写作辅助模块（模块六）：论文框架生成 / 拓扑图脚本 / 中译英 / 图注提炼 / DOI 解析
try:
    _writing = _load_module("writing_helper", "writing.py")
    generate_paper_framework = _writing.generate_paper_framework
    generate_diagram_scripts = _writing.generate_diagram_scripts
    translate_zh_to_en = _writing.translate_zh_to_en
    format_reference_catalog = _writing.format_reference_catalog
except Exception as e:
    print(f"[警告] 写作辅助模块加载失败（{e}），WRITE/DIAGRAM 分支不可用。")

    def generate_paper_framework(llm, idea, context_text, reference_catalog):
        raise RuntimeError("写作辅助模块未就绪")

    def generate_diagram_scripts(llm, description, formats=None):
        raise RuntimeError("写作辅助模块未就绪")

    def translate_zh_to_en(llm, text):
        raise RuntimeError("写作辅助模块未就绪")

    def format_reference_catalog(catalog):
        return ""


# ------------------------------------------------------------
# 定义 State（扩展字段以支撑 GRAPH / VISUAL 分支的数据传递）
# messages 使用 operator.add 做 reducer：各节点返回的消息会追加而非覆盖
# ------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str  # 路由节点写入的意图分类结果
    csv_path: Optional[str]  # VISUAL 分支: CSV 文件路径
    citations_data: Optional[list[dict[str, str]]]  # GRAPH 分支: 引用关系数据
    paper_id: Optional[str]  # RAG 分支: 指定论文时按 paper_id 过滤检索
    paper_ids: Optional[list[str]]  # RAG 分支: 跨论文问题时逐篇遍历检索
    search_query: Optional[str]  # RAG/TOOL 分支: LLM 改写后的英文检索词
    tool_results: Optional[dict[str, Any]]  # TOOL 分支: arXiv 检索/下载结构化结果
    lib_keywords: Optional[list[str]]  # TOOL 分支: 库内偏好关键词（推荐排序用）
    upload_dir: Optional[str]  # TOOL 分支: 下载 PDF 的保存目录
    plan_meta: Optional[dict[str, Any]]  # PLAN 分支: 子任务清单与执行元信息
    reference_catalog: Optional[
        list[dict[str, Any]]
    ]  # WRITE 分支: 库内真实文献清单（无幻觉 References 来源）
    write_meta: Optional[dict[str, Any]]  # WRITE 分支: 论文框架结构化产出与引用文献
    diagram_meta: Optional[dict[str, Any]]  # DIAGRAM 分支: 拓扑图脚本结构化产出


# 1. 意图分类节点 (Task Planner)
def router_node(state: AgentState):
    """分析用户最新一条消息，输出 RAG / GRAPH / VISUAL / WRITE / DIAGRAM / TOOL / PLAN 之一。"""
    last_message = state["messages"][-1].content
    prompt = f"""分析用户意图，仅返回以下选项之一：
    - 'RAG': 询问论文特定细节、超参数、硬件、特定出处（单一目标）；
    - 'GRAPH': 请求分析引用关系、领域基石节点（单一目标）；
    - 'VISUAL': 请求根据 CSV 生成图表（表格/折线/柱状/雷达图）；
    - 'WRITE': 请求根据 idea 生成论文框架（Abstract/Introduction/Related Work + References），或学术写作辅助；
    - 'DIAGRAM': 请求生成架构拓扑图脚本（TikZ / Graphviz / Mermaid / Matplotlib）；
    - 'TOOL': 请求联网检索 arXiv 论文、下载论文、查询领域最新论文或推荐相关论文；
    - 'PLAN': 复合任务——一次提问包含多个分析目标（如同时要求方法分析+引用图谱+综述生成+Future Work 探索）。

    用户输入: {last_message}
    分类结果:"""
    # 归一化：LLM 可能返回解释性文本，提取其中的意图标签，默认 RAG
    response = get_llm().invoke([HumanMessage(content=prompt)]).content.strip().upper()
    for tag in ("GRAPH", "VISUAL", "WRITE", "DIAGRAM", "TOOL", "PLAN", "RAG"):
        if tag in response:
            return {"next_step": tag}
    return {"next_step": "RAG"}


def _format_anchor_lines(chunks: list[dict[str, Any]]) -> str:
    """把检索锚点格式化为人类可读的溯源列表（序号 + 论文 + 页码 + 相关度）。

    bbox 坐标供前端芯片跳转高亮使用，正文中不再展示原始坐标。
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        pid = str(c.get("paper_id") or "")
        pid_short = pid[:28] + "…" if len(pid) > 28 else pid
        score = c.get("score")
        score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        loc = f"《{pid_short}》" if pid_short else ""
        lines.append(f"[{i}] {loc}第 {c['page_num']} 页 · 相关度 {score_txt}")
    return "\n".join(lines)


# 2. RAG 溯源回答节点（真实检索 + page_num/bbox 锚点，抗幻觉）
def rag_node(state: AgentState):
    """检索 Qdrant 中带锚点的文本块，要求 LLM 仅基于片段作答并标注来源页码。"""
    question = state["messages"][-1].content
    paper_id = state.get("paper_id")

    # 清洗查询：paper_id 来自上传文件名前缀，论文正文中并不存在，
    # 留在查询里会稀释语义、拉低检索相关度，嵌入前先剔除
    embed_query = question
    if paper_id:
        embed_query = embed_query.replace(paper_id, " ")
        # 文件名可能被截断引用（如 "1cvpr-2026-T-MOR-Lea"），逐段剥离前缀碎片
        for frag in paper_id.replace("_", " ").replace("-", " ").split():
            if len(frag) >= 4:
                embed_query = embed_query.replace(frag, " ")

    # 跨语言检索：本地哈希向量无跨语言语义能力，中文提问匹配英文论文时
    # 相关度趋近于 0；上游（app.py）若已用 LLM 把问题改写为英文检索词，
    # 则优先用改写结果做向量检索（回答仍针对原始问题）
    embed_query = state.get("search_query") or embed_query

    # ---- 多论文遍历路径：跨论文/全库问题时，逐篇过滤检索并分组拼装 ----
    paper_ids = state.get("paper_ids") or []
    if paper_ids:
        sections, anchors = [], []
        for pid in paper_ids:
            chunks = retrieve_with_anchors(embed_query, paper_id=pid, top_k=3)
            if not chunks:
                continue
            anchors.extend(chunks)
            body = "\n".join(f"[Page {c['page_num']}] {c['text']}" for c in chunks)
            sections.append(f"【论文：{pid}】\n{body}")
        if not sections:
            return {
                "messages": [AIMessage(content="未检索到相关论文内容，请先导入 PDF 或检查向量库。")]
            }
        context = "\n\n".join(sections)
        prompt = (
            f"你是严谨的科研助手。知识库已收录以下 {len(sections)} 篇论文的完整内容，"
            f"现按相关度给出每篇最相关的几处原文（标注页码）：\n{context}\n\n"
            f"请基于上述原文回答问题：{question}\n"
            f"要求：逐篇概括核心内容（各篇结论后以「根据论文第 X 页」标注依据），再做横向比较；"
            f"某篇原文不足以回答时明确说明，不要编造，也不要声称只掌握了论文片段。"
        )
        answer = get_llm().invoke([HumanMessage(content=prompt)]).content
        res = (
            f"{answer}\n\n———\n出处溯源（点击右侧锚点芯片可跳转原文高亮）：\n"
            f"{_format_anchor_lines(anchors)}"
        )
        return {"messages": [AIMessage(content=res)]}

    # ---- 单论文/全局路径 ----
    # 全文级提问（核心方法/总结/综述等）：top_k 5->10 扩大章节覆盖
    fulltext = bool(paper_id and FULLTEXT_RE.search(question))
    top_k = 10 if fulltext else 5
    chunks = retrieve_with_anchors(embed_query, paper_id=paper_id, top_k=top_k)
    if not chunks and paper_id:
        chunks = retrieve_with_anchors(embed_query, top_k=5)
    if not chunks:
        # 无检索结果时明确拒答，避免模型编造（抗幻觉兜底）
        return {
            "messages": [AIMessage(content="未检索到相关论文内容，请先导入 PDF 或检查向量库。")]
        }

    # 全文级提问：确定性注入摘要/引言块（首页直取，不依赖向量命中），
    # 摘要是方法概括的最关键依据，注入前按文本去重避免与检索结果重复
    if fulltext:
        existing = {c["text"] for c in chunks}
        head = [c for c in fetch_paper_head_chunks(paper_id) if c["text"] not in existing]
        chunks = head + chunks

    context = "\n".join(f"[Page {c['page_num']} | BBox{c['bbox']}] {c['text']}" for c in chunks)
    paper_txt = f"《{paper_id}》" if paper_id else "知识库中的论文"
    prompt = (
        f"你是严谨的科研助手。知识库已收录{paper_txt}的完整论文，"
        f"下面按相关度给出其中最相关的 {len(chunks)} 处原文（标注页码）：\n{context}\n\n"
        f"请基于上述原文回答问题：{question}\n"
        f"要求：以「根据论文第 X 页」的方式标注依据；"
        f"不要声称只掌握了论文的片段或部分内容——你看到的是完整论文中最相关的部分。"
    )
    answer = get_llm().invoke([HumanMessage(content=prompt)]).content
    # 附可读溯源列表；bbox/坐标由接口 anchors 字段传给前端芯片做跳转高亮
    res = (
        f"{answer}\n\n———\n出处溯源（点击右侧锚点芯片可跳转原文高亮）：\n"
        f"{_format_anchor_lines(chunks)}"
    )
    return {"messages": [AIMessage(content=res)]}


# 3. 图谱挖掘节点（PageRank + 介数中心性 挖基石论文）
def graph_node(state: AgentState):
    """对 state 中的引用关系数据跑图算法，输出基石论文与前端可用的图谱 JSON。"""
    data = state.get("citations_data") or []
    result = build_and_analyze_citation_graph(data)
    cornerstones = ", ".join(p for p, _ in result["cornerstones"]) or "无"
    g = result["graph_data"]
    # 边界防御：兜底函数异常时 graph_data 可能不是 dict，避免下标访问崩溃
    nodes = g["nodes"] if isinstance(g, dict) else []
    edges = g["edges"] if isinstance(g, dict) else []
    res = (
        f"引用图谱分析完成。基石论文：【{cornerstones}】。"
        f"图谱含 {len(nodes)} 个节点 / {len(edges)} 条边，"
        f"JSON 数据已就绪，可供前端 Cytoscape.js / Echarts 渲染。"
    )
    return {"messages": [AIMessage(content=res)]}


# 4. 可视化节点（CSV 智能绘图沙箱）
def visual_node(state: AgentState):
    """读取 CSV 并用 Matplotlib 生成趋势图，结果以 Base64 返回供前端直接渲染。"""
    csv_path = state.get("csv_path") or "data.csv"
    instruction = state["messages"][-1].content
    try:
        result = execute_data_visualization(csv_path, instruction)
        res = (
            f"图表已生成。{result['caption']}\n"
            f"(Base64 图像 {len(result['image_base64'])} 字符，供前端直接渲染)"
        )
    except Exception as e:
        res = f"绘图失败：{e}"
    return {"messages": [AIMessage(content=res)]}


# 4.5 学术写作节点（Idea -> 论文框架 + 无幻觉 References）
def write_node(state: AgentState):
    """根据用户 idea 生成结构化论文框架（Abstract/Intro/Related Work），
    Related Work 只允许引用库内真实文献，References 由 catalog 确定性拼装（无幻觉）。"""
    idea = state["messages"][-1].content
    catalog = state.get("reference_catalog") or []
    paper_ids = state.get("paper_ids") or []

    # ---- 1. 检索库内最相关的原文，用于支撑 Related Work 论点 ----
    # 有明确范围时遍历指定论文；否则在 catalog 覆盖的全库范围内检索。
    embed_query = state.get("search_query") or idea
    target_ids = paper_ids or [r.get("paper_id") for r in catalog]
    context_parts: list[str] = []
    cited_ids: list[str] = []
    for pid in target_ids:
        try:
            chunks = retrieve_with_anchors(embed_query, paper_id=pid, top_k=2)
        except Exception:
            chunks = []
        if not chunks:
            continue
        cited_ids.append(pid)
        body = "\n".join(f"[Page {c['page_num']}] {c['text']}" for c in chunks)
        title = next((r.get("title") or pid for r in catalog if r.get("paper_id") == pid), pid)
        context_parts.append(f"【{title}】\n{body}")

    # 检索不到时退回 catalog 前几篇（保证 References 非空，框架仍可生成）
    if not cited_ids:
        cited_ids = [r.get("paper_id") for r in catalog[:4]]
    context_text = "\n\n".join(context_parts) if context_parts else ""

    # ---- 2. 只把「实际引用」的文献传给 LLM，其余从 catalog 里按 cited_ids 摘取 ----
    cited_catalog = [r for r in catalog if r.get("paper_id") in cited_ids]
    # 保序去重
    seen = set()
    ordered: list = []
    for pid in cited_ids:
        if pid in seen:
            continue
        seen.add(pid)
        match = next((r for r in cited_catalog if r.get("paper_id") == pid), None)
        if match:
            ordered.append(match)

    try:
        framework = generate_paper_framework(get_llm(), idea, context_text, ordered)
    except Exception as e:
        framework = f"（论文框架生成失败：{e}）"

    # References 由上层（app.py）做 DOI 补全后确定性拼装，此处仅回传引用的文献清单。
    write_meta = {
        "idea": idea,
        "framework": framework,
        "references": [dict(r) for r in ordered],
        "cited_paper_ids": [r.get("paper_id") for r in ordered],
    }
    return {"messages": [AIMessage(content=framework)], "write_meta": write_meta}


# 4.6 架构拓扑图脚本生成节点（TikZ / Graphviz / Mermaid / Matplotlib）
def diagram_node(state: AgentState):
    """根据自然语言描述生成多格式架构拓扑图脚本。"""
    description = state["messages"][-1].content
    # 用户若点名了某一种格式，则只生成该格式；否则四种全给
    formats = []
    low = description.lower()
    if "tikz" in low or "latex" in low:
        formats.append("tikz")
    if "graphviz" in low or "dot" in low:
        formats.append("graphviz")
    if "mermaid" in low:
        formats.append("mermaid")
    if "matplotlib" in low or "python" in low or "py" in low:
        formats.append("matplotlib")
    formats = formats or None

    try:
        scripts = generate_diagram_scripts(get_llm(), description, formats)
    except Exception as e:
        scripts = {"error": str(e)}

    n = len(scripts)
    answer = (
        f"已生成 {n} 种格式的架构拓扑图脚本："
        + "、".join(
            f"{k}（{len(v.splitlines())} 行）" for k, v in scripts.items() if isinstance(v, str)
        )
        + "。脚本可在右侧面板查看 / 复制，Mermaid 可即时预览。"
    )
    return {"messages": [AIMessage(content=answer)], "diagram_meta": {"scripts": scripts}}


# 5. 工具调用节点（arXiv 检索 / 下载 / 推荐，闭环探索）
def tool_node(state: AgentState):
    """自主调用外部学术 API：arXiv 检索/最新论文 + GitHub 复现性 + 偏好推荐。

    模式判定：
      - 含"下载/导入" -> download：检索后自动下载 top1 到 upload_dir，
        由上层（app.py）完成入库，形成 搜索->下载->入库 闭环；
      - 含"推荐" -> recommend：库内偏好关键词与 GitHub stars 加权排序；
      - 其他 -> search：检索并附复现性信息。
    检索词优先用上游 LLM 改写后的英文 search_query（arXiv 为英文语料）。
    """
    import math
    import re as _re

    question = state["messages"][-1].content
    query = state.get("search_query") or question
    if _re.search(r"下载|导入|download", question, _re.I):
        mode = "download"
    elif _re.search(r"推荐|recommend", question, _re.I):
        mode = "recommend"
    else:
        mode = "search"

    try:
        if mode == "recommend":
            lib_kw = [w for w in (state.get("lib_keywords") or []) if len(w) > 2]
            fetch_q = " ".join(lib_kw[:4]) if lib_kw else query
            papers = arxiv_tool.search_arxiv(fetch_q, max_results=8, sort_by="submittedDate")
            kw_set = set(lib_kw)
            for p in papers:
                words = set(_re.findall(r"[a-z0-9]+", (p["title"] + " " + p["summary"]).lower()))
                try:
                    p["github"] = arxiv_tool.search_github(p["title"])
                except Exception:
                    p["github"] = None  # GitHub 限流/不可达时跳过复现性
                stars = p["github"]["stars"] if p["github"] else 0
                # 偏好重叠度为主，复现性（stars 取对数防碾压）为辅
                p["score"] = round(len(words & kw_set) * 2 + math.log1p(stars), 2)
            papers = sorted(papers, key=lambda x: -x["score"])[:5]
        else:
            papers = arxiv_tool.search_arxiv(query, max_results=5)
            for p in papers[:3]:  # GitHub 未认证限流 10 次/分钟，只查前 3
                try:
                    p["github"] = arxiv_tool.search_github(p["title"])
                except Exception:
                    p["github"] = None
    except Exception as e:
        return {
            "messages": [
                AIMessage(content=f"外部学术 API 调用失败（当前网络可能不可达 arXiv）：{e}")
            ],
            "tool_results": {"mode": mode, "papers": []},
        }

    # ---- 组装回答文本 ----
    lines = []
    for i, p in enumerate(papers, 1):
        gh = p.get("github")
        gh_txt = f"；复现仓库 {gh['repo']}（★{gh['stars']}）" if gh else ""
        lines.append(
            f"[{i}] {p['title']}（{p['published']}，arXiv:{p['arxiv_id']}{gh_txt}）\n"
            f"    作者：{', '.join(p['authors'][:3])}{' 等' if len(p['authors']) > 3 else ''}\n"
            f"    摘要：{p['summary'][:120]}…"
        )
    header = {
        "search": f"arXiv 检索「{query}」命中 {len(papers)} 篇：",
        "recommend": f"结合库内偏好与 GitHub 复现性，推荐 {len(papers)} 篇最新论文：",
        "download": f"arXiv 检索「{query}」命中 {len(papers)} 篇，已自动下载第 1 篇：",
    }[mode]
    answer = header + "\n" + "\n".join(lines) if papers else f"未检索到相关论文（查询：{query}）。"

    result = {"mode": mode, "papers": papers}
    # ---- download 模式：自动下载 top1，交由上层入库（闭环） ----
    if mode == "download" and papers:
        top = papers[0]
        upload_dir = state.get("upload_dir") or "."
        safe_name = _re.sub(r'[\\/:*?"<>|]', "_", top["title"])[:80] or top["arxiv_id"]
        save_path = os.path.join(upload_dir, f"{safe_name}.pdf")
        try:
            arxiv_tool.download_pdf(top["pdf_url"], save_path)
            result["downloaded_file"] = save_path
            result["downloaded_title"] = top["title"]
            answer += f"\n\n已下载：{top['title']} -> {os.path.basename(save_path)}（正在自动解析入库...）"
        except Exception as e:
            answer += f"\n\n下载失败：{e}"
    return {"messages": [AIMessage(content=answer)], "tool_results": result}


# 6. PLAN 复合任务编排：多目标提问拆解为子任务，依次执行后合并
def _survey_section(state: AgentState, embed_query: str, paper_ids: list[str]) -> str:
    """领域综述子任务：多论文检索原文 -> LLM 撰写结构化迷你综述。"""
    sections = []
    for pid in paper_ids:
        chunks = retrieve_with_anchors(embed_query, paper_id=pid, top_k=3)
        if chunks:
            body = "\n".join(f"[Page {c['page_num']}] {c['text']}" for c in chunks)
            sections.append(f"【论文：{pid}】\n{body}")
    if not sections:
        return "知识库中无可用于综述的论文内容。"
    prompt = (
        f"基于以下 {len(sections)} 篇论文的原文内容，撰写一段领域综述：\n\n"
        + "\n\n".join(sections)
        + "\n\n要求：中文；结构为 研究背景 -> 各方法技术路线对比 -> 发展趋势；"
        "关键论断标注来源（论文名 + 页码）；不超过 600 字。"
    )
    return get_llm().invoke([HumanMessage(content=prompt)]).content


def _future_section(state: AgentState, paper_ids: list[str]) -> str:
    """Future Work 子任务：逐篇检索未来工作/局限相关原文 -> 结合时间线分析研究空白。"""
    fw_query = "future work limitations research directions conclusion"
    sections = []
    for pid in paper_ids:
        chunks = retrieve_with_anchors(fw_query, paper_id=pid, top_k=3)
        if chunks:
            body = "\n".join(f"[Page {c['page_num']}] {c['text']}" for c in chunks)
            sections.append(f"【论文：{pid}】\n{body}")
    if not sections:
        return "未检索到各论文的 Future Work / 局限性相关内容。"
    prompt = (
        "以下是多篇论文（按发表时间先后排列）的原文内容：\n\n"
        + "\n\n".join(sections)
        + "\n\n请完成：1) 逐篇提取其中提到的 Future Work、局限性或未解决问题；"
        "2) 结合论文时间线，分析哪些早期 Idea 已被后续工作解决（指出解决者），"
        "哪些至今仍是研究空白；3) 输出最值得跟进的 2-3 个研究方向。"
        "中文回答，标注来源页码。"
    )
    return get_llm().invoke([HumanMessage(content=prompt)]).content


def plan_node(state: AgentState):
    """复合任务编排节点：识别提问中的多个分析目标，依次调用各能力后合并结果。

    子任务判定（关键词确定性拆解，保证可解释、可复现）：
      RAG    - 核心方法 / 异同 / 对比分析（复用 rag_node，多论文时走遍历路径）；
      GRAPH  - 引用关系 / 图谱 / 基石挖掘（复用 graph_node）；
      SURVEY - 领域综述生成；
      FUTURE - Future Work 提取 + 结合时间线的研究空白分析。
    """
    question = state["messages"][-1].content
    paper_ids = state.get("paper_ids") or []
    embed_query = state.get("search_query") or question

    tasks = []
    if re.search(r"核心方法|异同|对比|差异|分析.*论文|方法.*比较", question):
        tasks.append("RAG")
    if re.search(r"引用|图谱|基石|拓扑|局域", question):
        tasks.append("GRAPH")
    if re.search(r"综述", question):
        tasks.append("SURVEY")
    if re.search(r"Future Work|未来|研究空白|展望|趋势", question, re.I):
        tasks.append("FUTURE")
    if not tasks:  # 未识别到明确子任务时退化为单论文/全库问答
        tasks = ["RAG"]

    titles = {
        "RAG": "核心方法分析",
        "GRAPH": "引用关系与基石挖掘",
        "SURVEY": "领域综述",
        "FUTURE": "Future Work 与研究空白",
    }
    cn = {"RAG": "一", "GRAPH": "二", "SURVEY": "三", "FUTURE": "四"}

    sections = [
        f"已识别为复合任务，拆解为 {len(tasks)} 个子任务依次执行："
        f"{' → '.join(titles[t] for t in tasks)}。\n"
    ]
    for t in tasks:
        if t == "RAG":
            out = rag_node(state)  # paper_ids 已由上层按全库/点名注入，自动走遍历路径
            sections.append(f"## {cn[t]}、{titles[t]}\n{out['messages'][-1].content}")
        elif t == "GRAPH":
            out = graph_node(state)
            sections.append(f"## {cn[t]}、{titles[t]}\n{out['messages'][-1].content}")
        elif t == "SURVEY":
            sections.append(
                f"## {cn[t]}、{titles[t]}\n" f"{_survey_section(state, embed_query, paper_ids)}"
            )
        elif t == "FUTURE":
            sections.append(f"## {cn[t]}、{titles[t]}\n" f"{_future_section(state, paper_ids)}")

    meta = {"tasks": tasks, "has_graph": "GRAPH" in tasks}
    return {"messages": [AIMessage(content="\n\n".join(sections))], "plan_meta": meta}


# 7. 路由分支判断：把 router 的文本结论映射到具体节点名
def decide_next(state: AgentState):
    step = state.get("next_step", "RAG")
    if "PLAN" in step:
        return "plan_node"
    if "GRAPH" in step:
        return "graph_node"
    elif "VISUAL" in step:
        return "visual_node"
    elif "WRITE" in step:
        return "write_node"
    elif "DIAGRAM" in step:
        return "diagram_node"
    elif "TOOL" in step:
        return "tool_node"
    return "rag_node"


# 8. 构建 LangGraph 工作流
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("graph_node", graph_node)
workflow.add_node("visual_node", visual_node)
workflow.add_node("write_node", write_node)
workflow.add_node("diagram_node", diagram_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("plan_node", plan_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    decide_next,
    {
        "rag_node": "rag_node",
        "graph_node": "graph_node",
        "visual_node": "visual_node",
        "write_node": "write_node",
        "diagram_node": "diagram_node",
        "tool_node": "tool_node",
        "plan_node": "plan_node",
    },
)
for node in (
    "rag_node",
    "graph_node",
    "visual_node",
    "write_node",
    "diagram_node",
    "tool_node",
    "plan_node",
):
    workflow.add_edge(node, END)

# checkpointer：状态持久化，支持任务打断与恢复；
# interrupt_before=["graph_node"]：Human-in-the-Loop —— 图谱分析前暂停，
# 等待人工确认（如确认语义去重结果、确认引用数据无误）后再继续执行。
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["graph_node"],
)


# ============================================================
# 演示入口：三条测试用例覆盖 RAG / GRAPH(含人工打断确认) / VISUAL 三分支
# ============================================================
if __name__ == "__main__":
    if USE_MOCK_LLM:
        print("[提示] 未检测到 OPENAI_API_KEY，当前使用离线 Mock LLM 演示流程。\n")

    # thread_id 标识一次会话，checkpointer 依此存取状态（支持打断恢复）
    thread = {"configurable": {"thread_id": "demo-1"}}

    # ---- 用例 1：RAG 溯源问答 ----
    print("=== 用例1: RAG 溯源问答 ===")
    out = app.invoke(
        {"messages": [HumanMessage(content="这篇论文训练用了什么硬件？")]},
        config=thread,
    )
    print(out["messages"][-1].content, "\n")

    # ---- 用例 2：GRAPH 图谱分析（演示 Human-in-the-Loop 打断与确认恢复） ----
    print("=== 用例2: 引用图谱分析（HITL 打断演示） ===")
    out = app.invoke(
        {
            "messages": [HumanMessage(content="帮我分析这些论文的引用图谱，找出基石论文")],
            "citations_data": [
                {"source": "BERT_2018", "target": "Transformer_2017"},
                {"source": "GPT1_2018", "target": "Transformer_2017"},
                {"source": "ViT_2020", "target": "Transformer_2017"},
                {"source": "RoBERTa_2019", "target": "BERT_2018"},
            ],
        },
        config=thread,
    )
    # 因在 graph_node 前设置了 interrupt_before，此处流程处于暂停状态
    snapshot = app.get_state(thread)
    if snapshot.next:
        print(f"[HITL] 流程已在 {snapshot.next} 前暂停，等待人工确认引用数据...")
        print("[HITL] 人工确认无误，恢复执行。")
        out = app.invoke(None, config=thread)  # 传 None 表示从断点继续
    print(out["messages"][-1].content, "\n")

    # ---- 用例 3：VISUAL CSV 智能绘图 ----
    print("=== 用例3: CSV 智能绘图 ===")
    sample_csv = os.path.join(ROOT_DIR, "data", "data.csv")
    if not os.path.exists(sample_csv):
        # 生成一份示例训练日志 CSV（Epoch/Accuracy），供绘图演示
        with open(sample_csv, "w", encoding="utf-8") as f:
            f.write("Epoch,Accuracy\n")
            for epoch, acc in enumerate([55, 63, 71, 77, 82, 86, 89, 91, 92, 93], start=1):
                f.write(f"{epoch},{acc}\n")
        print(f"[提示] 已生成示例数据 {sample_csv}")
    out = app.invoke(
        {
            "messages": [HumanMessage(content="请根据 CSV 画出 Accuracy 的训练曲线图")],
            "csv_path": sample_csv,
        },
        config=thread,
    )
    print(out["messages"][-1].content)
