# writing.py
# ============================================================
# 学术写作辅助模块（模块六核心能力）
#
# 职责：把「写作 / 翻译 / 图注 / 文献 DOI」相关的 LLM 调用收敛到一处，
#       供 agent.py（write_node / diagram_node）与 front_window/app.py 复用。
#
# 设计约定：
#   - 本模块不 import agent.py，避免循环依赖；所有函数显式接收 `llm`
#     （一个带 .invoke([HumanMessage(...)]) 的对象，即 get_llm() 的返回值）。
#   - 网络请求复用 arxiv_tool._get（带代理 / 证书降级），保证离线降级不抛。
# ============================================================
from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any

import arxiv_tool  # 复用其带代理的 _get

# ---- Crossref 标题检索（文献 DOI 补全） ----
CROSSREF_QUERY = "https://api.crossref.org/works"


def resolve_doi_by_title(title: str, authors: str = "") -> dict[str, Any]:
    """按论文标题到 Crossref 检索，补全 DOI / 作者 / 期刊 / 年份。

    优先「bibliographic 检索 + 标题相似度」判定是否同一篇，避免同名误配；
    失败或离线时返回空 dict，调用方据此降级（References 仅展示题名，不写 DOI）。
    """
    title = (title or "").strip()
    if not title:
        return {}
    # 去掉 arXiv 版本号后缀（v1/v2），避免检索不到
    probe = re.sub(r"v\d+$", "", title).strip() or title
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": probe,
            "rows": 1,
            "select": "DOI,title,author,container-title,issued",
        }
    )
    try:
        data = json.loads(
            arxiv_tool._get(f"{CROSSREF_QUERY}?{params}", timeout=12).decode("utf-8", "ignore")
        )
        items = (data.get("message") or {}).get("items") or []
    except Exception:
        return {}
    if not items:
        return {}
    it = items[0]
    hit_title = ((it.get("title") or [""])[0]).strip()
    if not _title_sim(title, hit_title) and not _title_sim(probe, hit_title):
        return {}
    authors_list = [
        f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (it.get("author") or [])
    ][:8]
    authors_list = [a for a in authors_list if a]
    issued = (it.get("issued") or {}).get("date-parts") or [[""]]
    year = str(issued[0][0]) if issued and issued[0] else ""
    return {
        "doi": it.get("DOI", ""),
        "title": hit_title,
        "authors": authors_list,
        "journal": ((it.get("container-title") or [""])[0]).strip(),
        "year": year,
    }


def _title_sim(a: str, b: str) -> bool:
    """标题词集重合度粗判（区分同名论文）。"""
    aw = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    bw = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not aw:
        return False
    return len(aw & bw) / len(aw) >= 0.6


# ---- 论文框架生成（Idea -> Abstract/Intro/Related Work） ----
def format_reference_catalog(catalog: list[dict[str, Any]]) -> str:
    """把知识库文献清单格式化为编号列表，供 LLM 在 Related Work 中按编号引用。"""
    lines = []
    for i, ref in enumerate(catalog or [], 1):
        authors = ", ".join(ref.get("authors") or [])
        year = ref.get("year") or ""
        title = ref.get("title") or ref.get("paper_id") or ""
        meta = f"{authors} ({year}). {title}." if authors else title
        if ref.get("doi"):
            meta += f" DOI: {ref['doi']}"
        lines.append(f"[{i}] {meta}")
    return "\n".join(lines)


def generate_paper_framework(
    llm: Any,
    idea: str,
    context_text: str,
    reference_catalog: list[dict[str, Any]],
) -> str:
    """根据 idea 生成结构化论文框架（Abstract / Introduction / Related Work）。

    Related Work 只允许引用 reference_catalog 中编号的真实文献，
    从源头上避免幻觉引用；References 列表由调用方（app.py）确定性拼装。
    """
    ref_block = format_reference_catalog(reference_catalog)
    ctx = context_text or "（知识库未检索到相关原文，可仅基于 idea 与文献标题撰写。）"
    prompt = (
        "你是资深顶会论文写作助手。请基于用户的 idea 与知识库真实文献，"
        "撰写一份可直接投顶会的英文论文框架，严格按下面 Markdown 结构输出：\n\n"
        "# Title\n（一个信息量足、学术风格的英文标题）\n\n"
        "## Abstract\n（150 词左右，问题 -> 方法 -> 关键结果 -> 贡献）\n\n"
        "## Introduction\n（研究背景与动机、现有方法不足、本文贡献 3 点，用 [n] 标注引用）\n\n"
        "## Related Work\n（按主题分 2~3 小段梳理，与 idea 最相关的方向重点展开，"
        "每处论断都要用 [n] 指向下面的真实文献）\n\n"
        "## References\n（这里只输出编号占位，例如 [1][2][3]，正文由系统按文献清单自动补全）\n\n"
        "硬性要求：\n"
        "1. 引用标记 [n] 只能使用下面「真实文献清单」里的编号，严禁引用清单外文献、严禁编造 DOI；\n"
        "2. 全文使用英文；语气客观、用词符合学术规范；\n"
        "3. 不要在正文里写没有依据的具体数值或实验结论。\n\n"
        f"用户 idea：\n{idea}\n\n"
        f"知识库相关原文（用于支撑 Related Work 论点）：\n{ctx}\n\n"
        f"真实文献清单（可引用的唯一来源）：\n{ref_block if ref_block else '（知识库为空，References 留空）'}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def _extract_sections(framework: str) -> dict[str, str]:
    """把 Markdown 框架按一级/二级标题切分为有序段落，供前端分块卡片渲染。"""
    sections: list[dict[str, str]] = []
    cur_title = "Preamble"
    cur_body: list[str] = []
    for line in framework.splitlines():
        if re.match(r"^#{1,2}\s+", line):
            if cur_body or cur_title:
                sections.append({"title": cur_title, "body": "\n".join(cur_body).strip()})
            cur_title = re.sub(r"^#{1,2}\s+", "", line).strip()
            cur_body = []
        else:
            cur_body.append(line)
    if cur_title or cur_body:
        sections.append({"title": cur_title, "body": "\n".join(cur_body).strip()})
    # 去掉空段落
    sections = [s for s in sections if s["body"]]
    out: dict[str, str] = {}
    for s in sections:
        out[s["title"]] = s["body"]
    return out


# ---- 架构拓扑图脚本生成（TikZ / Graphviz / Mermaid / Matplotlib） ----
DIAGRAM_FORMATS = ["tikz", "graphviz", "mermaid", "matplotlib"]

_DIAGRAM_FORMAT_HINTS = {
    "tikz": "LaTeX TikZ（\\documentclass[tikz]{standalone}，节点/边/标签，可直接编译）",
    "graphviz": "Graphviz DOT（digraph，节点、边、子图 cluster）",
    "mermaid": "Mermaid（graph TD/LR 流程图，可直接粘贴到支持 Mermaid 的渲染器）",
    "matplotlib": "Python + matplotlib（networkx 布局或手绘坐标，输出 png 的完整脚本）",
}


def generate_diagram_scripts(
    llm: Any, description: str, formats: list[str] | None = None
) -> dict[str, str]:
    """根据自然语言描述生成多种格式的架构拓扑图脚本。

    返回 {"tikz": "...", "graphviz": "...", "mermaid": "...", "matplotlib": "..."}，
    用代码围栏包裹以便前端按 fenced code block 解析。
    """
    formats = formats or DIAGRAM_FORMATS
    out: dict[str, str] = {}
    for fmt in formats:
        hint = _DIAGRAM_FORMAT_HINTS.get(fmt, fmt)
        prompt = (
            f"你是科研绘图专家。请根据下面的架构/流程描述，生成一段 {hint} 的脚本。\n"
            "要求：\n"
            "1. 只输出代码，包在一个 ``` 代码围栏里（围栏语言标签写为对应格式名），不要任何额外解释；\n"
            "2. 拓扑结构清晰、命名专业、可读性好；给关键节点写注释；\n"
            "3. 不要出现占位符 TODO，也不要输出任何需要用户再补充的部分。\n\n"
            f"架构/流程描述：\n{description}"
        )
        try:
            text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception:
            text = ""
        out[fmt] = _strip_fence(text)
    return out


def _strip_fence(text: str) -> str:
    """剥掉 LLM 输出最外层的一对 ``` 围栏（若存在），返回纯代码。"""
    text = (text or "").strip()
    m = re.match(r"^```[^\n]*\n(.*?)\n?```$", text, re.DOTALL)
    return m.group(1).strip() if m else text


# ---- 中译英学术翻译 ----
def translate_zh_to_en(llm: Any, text: str) -> str:
    """把中文论文摘要 / 全文翻译为符合学术规范的英文。

    保留公式 / LaTeX / 变量名 / 数字，术语与语气符合期刊写作习惯。
    """
    prompt = (
        "你是学术翻译专家。请把下面的中文论文内容翻译成符合国际期刊规范的英文。\n"
        "要求：\n"
        "1. 语气客观、术语准确、句式地道，避免中式英语；\n"
        "2. 公式、LaTeX、变量名、数字、单位、参考文献编号（如 [12]）一律保留不译；\n"
        "3. 只输出译文，不要解释、不要附原文；\n"
        "4. 若原文是完整段落，译文同样按段落输出；标题层级用对应 Markdown 标题保留。\n\n"
        f"待翻译内容：\n{text}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


# ---- Figure Caption 自适应提炼 ----
def generate_figure_caption(llm: Any, stats_text: str, chart_type: str, instruction: str) -> str:
    """根据数据统计与图表类型，提炼符合学术规范、能说明核心趋势的 Figure Caption。"""
    prompt = (
        "你是数据可视化专家。请为下面的图表撰写一条学术规范的 Figure Caption（英文）。\n"
        "要求：\n"
        "1. 以 'Figure 1: ' 开头，一句话点明图表呈现的核心趋势或结论（不要罗列数据）；\n"
        "2. 风格参考顶会论文图注（客观、精炼，20~60 词）；\n"
        "3. 只输出 caption 本身，不要额外解释。\n\n"
        f"图表类型：{chart_type}\n"
        f"用户绘图意图：{instruction}\n"
        f"数据统计摘要：\n{stats_text}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def build_stats_text(df_summary: dict[str, Any]) -> str:
    """把可视化模块统计出的数值摘要拼成自然语言，供 caption LLM 使用。"""
    lines = []
    for col, info in (df_summary or {}).items():
        if not isinstance(info, dict):
            continue
        parts = [f"{col}: "]
        for k in ("count", "mean", "std", "min", "max", "trend", "correlation"):
            if k in info and info[k] != "":
                parts.append(f"{k}={info[k]}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


# 兼容导入：让 agent.py 的 `from langchain_core.messages import HumanMessage` 之外，
# 此处也仅做类型占位——实际调用方传入 HumanMessage 即可。
try:
    from langchain_core.messages import HumanMessage  # pyright: ignore[reportMissingImports]
except Exception:  # noqa: BLE001 —— 离线/未装依赖时不阻断模块导入
    HumanMessage = object  # type: ignore[assignment,misc]
