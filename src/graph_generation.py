# graph_generation.py
# 延迟注解求值：允许 list[...] 等新式泛型在低版本分析目标下通过静态检查
from __future__ import annotations

from typing import Any

# pyright 定向抑制：IDE 分析环境解释器未装依赖时误报，运行时依赖真实存在
import networkx as nx  # pyright: ignore[reportMissingModuleSource]


def build_and_analyze_citation_graph(citations_data: list[dict[str, str]], top_k: int = 2):
    """
    citations_data 结构示例:
    [
      {"source": "Paper_A", "target": "Attention_1706.03762"},
      {"source": "Paper_B", "target": "Attention_1706.03762"},
      {"source": "Paper_C", "target": "Paper_A"}
    ]

    通过 PageRank + 介数中心性(Betweenness Centrality) 双指标
    挖掘领域基石论文，并导出前端 Cytoscape.js / Echarts 可用的 JSON。
    """
    G = nx.DiGraph()

    for item in citations_data:
        G.add_edge(item["source"], item["target"])

    # 边界处理：空图直接返回，避免 pagerank 报错
    if G.number_of_nodes() == 0:
        return {"cornerstones": [], "graph_data": {"nodes": [], "edges": []}}

    # PageRank：被高价值论文引用的论文更重要（找"开山之作"）
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    # 介数中心性：处于多篇论文引用路径交汇处的"桥梁"论文
    betweenness = nx.betweenness_centrality(G)

    # 双指标综合得分，取 Top-K 基石论文
    combined = {n: pagerank_scores[n] + betweenness.get(n, 0.0) for n in G.nodes()}
    sorted_papers = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    cornerstone_papers = sorted_papers[:top_k]

    # 导出为前端 Cytoscape.js / Echarts 可用的 JSON 格式
    graph_json = {
        "nodes": [
            {
                "id": node,
                "label": node,
                "val": round(pagerank_scores[node] * 100, 2),
                "betweenness": round(betweenness.get(node, 0.0), 4),
            }
            for node in G.nodes()
        ],
        "edges": [{"source": edge[0], "target": edge[1]} for edge in G.edges()],
    }

    return {
        "cornerstones": cornerstone_papers,
        "graph_data": graph_json,
    }


# ============================================================
# 真实引用网络抽取：从论文参考文献区解析条目并建边
# ============================================================
def parse_reference_entries(ref_text: str) -> list[str]:
    """把参考文献区拼接文本按 [n] 编号切分为单条文献条目。

    PyMuPDF 提取的参考文献块常一条跨多块/一块多条，先整体拼接再切分。
    """
    import re

    parts = re.split(r"(?=\[\d+\])", ref_text)
    return [p.strip() for p in parts if re.match(r"^\[\d+\]", p.strip()) and len(p.strip()) > 25]


def extract_ref_title(entry: str, max_len: int = 60) -> str:
    """从单条文献条目提取标题（CVPR 格式：作者. 标题. 出处）。

    启发式：作者段含 "," 或 " and " 或为缩写碎片则跳过，
    其后第一个像标题的段落作为标题；解析失败退回条目截断。
    """
    import re

    body = re.sub(r"^\[\d+\]\s*", "", entry).strip()
    segs = [s.strip() for s in body.split(". ") if s.strip()]
    idx = 0
    while idx < len(segs) - 1:
        s = segs[idx]
        # 作者区特征：含逗号（"Smith, A."）、含 and（多作者）、纯缩写碎片（"A."）
        if ("," in s) or (" and " in s.lower()) or re.fullmatch(r"([A-Z]\.?\s*)+", s):
            idx += 1
            continue
        # 过短的非作者段视为缩写碎片，继续跳过
        if len(s.split()) <= 2:
            idx += 1
            continue
        break
    if idx < len(segs):
        title = re.split(r"\b(In Proceedings|In Proc|arXiv|IEEE|ACM|Advances in)\b", segs[idx])[
            0
        ].strip(" .-")
        if len(title.split()) >= 2:
            return title[:max_len]
    return body[:max_len]


def _norm_words(text: str) -> set[str]:
    """归一化词集：小写、去连字符断行、去停用词与纯数字，用于相似匹配。"""
    import re

    text = re.sub(r"-\s+", "", text.lower())  # 修复跨行连字符断词
    words = set(re.findall(r"[a-z0-9]+", text))
    stop = {"the", "a", "an", "of", "in", "and", "for", "with", "on", "to", "via", "by"}
    return {w for w in words if w not in stop and not w.isdigit() and len(w) > 1}


def build_citation_edges(
    refs_by_paper: dict[str, str],
    lib_labels: dict[str, str],
    min_shared: int = 2,
    max_external: int = 25,
    match_ratio: float = 0.6,
) -> list[dict[str, str]]:
    """
    根据各论文参考文献文本构建真实引用边。

    refs_by_paper: {论文标签: 该论文参考文献区拼接文本}
    lib_labels:    {paper_id: 库内论文显示标签}（用于判定"引了库内论文"）

    建边规则：
      1. 条目标题词集与某库内论文标题词集重叠率 >= match_ratio -> 库内引用边；
      2. 否则为外部文献：按词集 Jaccard >= 0.6 聚类合并（不同论文的参考文献
         对同一工作的条目文字略有差异，聚类实现"共引合并"）；
      3. 被引次数 >= min_shared 的外部文献生成外部节点边——
         共引越多 PageRank 越高，正是要挖的"领域基石"；
         若无人达标，按被引次数补足至多 max_external 个。

    返回 [{source, target}, ...]，可直接喂给 build_and_analyze_citation_graph。
    """
    edges = []
    lib_title_words = {pid: _norm_words(label) for pid, label in lib_labels.items()}
    label_by_pid = dict(lib_labels)

    # 外部文献聚类：[{"words": set, "label": str, "sources": set}]
    clusters = []

    for paper_label, ref_text in refs_by_paper.items():
        for entry in parse_reference_entries(ref_text):
            entry_words = _norm_words(entry)
            if not entry_words:
                continue
            # ---- 1. 是否引用了库内论文 ----
            hit_pid = None
            for pid, tw in lib_title_words.items():
                if not tw or label_by_pid[pid] == paper_label:
                    continue  # 不计自引
                overlap = len(entry_words & tw) / len(tw)
                if overlap >= match_ratio:
                    hit_pid = pid
                    break
            if hit_pid:
                edges.append({"source": paper_label, "target": label_by_pid[hit_pid]})
                continue
            # ---- 2. 外部文献：共引聚类 ----
            title = extract_ref_title(entry)
            for c in clusters:
                inter = len(entry_words & c["words"])
                union = len(entry_words | c["words"]) or 1
                if inter / union >= 0.6:
                    c["sources"].add(paper_label)
                    c["words"] |= entry_words
                    if len(title) > len(c["label"]):
                        c["label"] = title
                    break
            else:
                clusters.append(
                    {"words": set(entry_words), "label": title, "sources": {paper_label}}
                )

    # ---- 3. 共引外部文献生成边 ----
    clusters.sort(key=lambda c: len(c["sources"]), reverse=True)
    chosen = [c for c in clusters if len(c["sources"]) >= min_shared]
    if not chosen:
        chosen = clusters[:max_external]
    for c in chosen[:max_external]:
        for src in c["sources"]:
            edges.append({"source": src, "target": c["label"]})

    # 去重（同一 source->target 只留一条）
    seen, uniq = set(), []
    for e in edges:
        key = (e["source"], e["target"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq


def extract_future_work_and_gap(papers: list[dict[str, Any]], llm=None, sim_threshold: float = 0.6):
    """
    Future Work 抽取与 Research Gap（研究空白）分析。

    对应 PDF 方案一第二阶段「用 Prompt + 图谱抽取 Future Work，
    实现简单的 Research Gap 分析」：提取各论文的 Future Work，
    结合最新论文时间线自动剔除已被解决的旧 Idea。

    papers 结构示例（按发表时间升序）:
    [
      {"id": "Paper_A", "year": 2018,
       "future_work": "提升长文本外推能力",
       "abstract": "本文提出一种相对位置编码，解决了长文本外推问题。"},
      ...
    ]

    逻辑：
      1. 汇总所有论文的 Future Work 条目，并用文本相似度去重
         （同一研究方向被多篇论文反复提及只保留一次）；
      2. 对每条 Future Work，检查发表更晚论文的 abstract 是否已覆盖该方向
         （关键词重叠判定；接入 llm 后可替换为 LLM 语义判断）；
         被覆盖则标记 solved=True，视为"已被解决的旧 Idea"予以过滤；
      3. 剩余 solved=False 的条目即为候选 Research Gap。

    返回: {"gaps": [...], "solved": [...]}，每项含来源论文与判定理由。
    """
    from difflib import SequenceMatcher

    # ---- 第 1 步：抽取全部 Future Work 并语义去重 ----
    fw_items = []  # {"idea": 文本, "source": 论文id, "year": 年份}
    for p in papers:
        idea = (p.get("future_work") or "").strip()
        if not idea:
            continue
        # 与已收集的条目比对，相似度超过阈值视为同一研究方向，跳过去重
        if any(SequenceMatcher(None, idea, it["idea"]).ratio() >= sim_threshold for it in fw_items):
            continue
        fw_items.append({"idea": idea, "source": p.get("id", "?"), "year": p.get("year", 0)})

    # ---- 第 2 步：结合时间线剔除已被解决的旧 Idea ----
    gaps, solved = [], []
    for it in fw_items:
        idea_keywords = set(it["idea"])  # 字符级集合，中英文均可粗略重叠匹配
        solved_by, reason = None, ""
        for p in papers:
            # 只考察比该 Future Work 提出时间更晚的论文
            if p.get("year", 0) <= it["year"]:
                continue
            text = f"{p.get('title', '')} {p.get('abstract', '')}"
            if llm is not None:
                # 接入真实 LLM 时：用语义判断替代关键词重叠
                verdict = llm.invoke(
                    f"判断以下论文是否已解决该研究问题：'{it['idea']}'。\n"
                    f"论文摘要：{text}\n仅回答 '是' 或 '否'。"
                ).content.strip()
                hit = "是" in verdict
            else:
                # 离线规则兜底：摘要与 Idea 的字符重叠率 > 30% 视为已覆盖
                overlap = len(idea_keywords & set(text)) / max(len(idea_keywords), 1)
                hit = overlap > 0.3
            if hit:
                solved_by, reason = p.get("id", "?"), "后续论文已覆盖该方向"
                break
        record = {**it, "solved": bool(solved_by), "solved_by": solved_by, "reason": reason}
        (solved if solved_by else gaps).append(record)

    return {"gaps": gaps, "solved": solved}


# 测试数据
if __name__ == "__main__":
    test_citations = [
        {"source": "BERT_2018", "target": "Transformer_2017"},
        {"source": "GPT1_2018", "target": "Transformer_2017"},
        {"source": "ViT_2020", "target": "Transformer_2017"},
        {"source": "RoBERTa_2019", "target": "BERT_2018"},
    ]

    result = build_and_analyze_citation_graph(test_citations)
    print("基石论文挖掘结果:", result["cornerstones"])
    print("图谱 JSON:", result["graph_data"])

    # ---- Future Work / Research Gap 分析演示 ----
    # 模拟 3 篇论文（按年份升序）：Paper_A 提出的 Idea 被 Paper_C 解决，
    # Paper_B 提出的 Idea 尚未被解决 -> 应被识别为 Research Gap
    test_papers = [
        {
            "id": "Paper_A",
            "year": 2018,
            "future_work": "提升长文本外推能力",
            "abstract": "本文提出一种基础编码结构。",
        },
        {
            "id": "Paper_B",
            "year": 2019,
            "future_work": "降低多模态对齐的计算开销",
            "abstract": "本文研究多模态特征融合。",
        },
        {
            "id": "Paper_C",
            "year": 2020,
            "future_work": "",
            "abstract": "本文通过旋转位置编码提升长文本外推能力，效果显著。",
        },
    ]
    gap_result = extract_future_work_and_gap(test_papers)
    print("\n候选 Research Gap（未被解决的 Idea）:")
    for g in gap_result["gaps"]:
        print(f"  - [{g['source']} {g['year']}] {g['idea']}")
    print("已被解决的旧 Idea:")
    for s in gap_result["solved"]:
        print(f"  - [{s['source']}] {s['idea']} -> 由 {s['solved_by']} 解决")
