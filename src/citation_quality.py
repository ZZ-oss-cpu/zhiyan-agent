# citation_quality.py
# ============================================================
# 参考文献「引用含金量」评估与筛选（对应赛题模块四：自动剔除应付式引用）
#
# 设计目标：
#   1) **可解释**：每条文献给出 0-100 分 + 逐项得分 + 中文理由，评审能追问每一项；
#   2) **可离线**：全部为确定性规则（正则 / 词法 / 结构信号），不依赖 LLM 与网络，
#      答辩断网也能跑；联网能力（Crossref 核验）由调用方按需补充，不绑进主链路；
#   3) **可复用**：入口既接受 RIS / BibTeX / 纯文本参考文献，也接受已解析条目。
#
# 五个信号（满分 100）：
#   A. 元数据完整度 25  —— 作者 / 年份 / 发表venue / DOI·arXiv 标识
#   B. 可核验性     20  —— DOI 格式合法、arXiv ID、可信 URL
#   C. 发表形态     15  —— 期刊/会议（同行评议）> 预印本 > 未知
#   D. 使用强度     25  —— 正文中被真实引用的次数（无正文时给中性 12 分）
#   E. 时效性       10  —— 近 3 年 / 10 年 / 20 年
# 扣分项：重复条目 -30、占位/无题名 -20、标题过短 -12、年份异常 -10
#
# 判定档位（阈值默认 50）：
#   >=70 核心（high）  50-69 支撑（good）  30-49 边缘（weak）  <30 存疑（drop）
# ============================================================
from __future__ import annotations

import re
from typing import Any

import bibparse

# 各信号满分（改权重只需改这里，前端展示按信号名读取）
MAX_SCORE = {
    "metadata": 25,
    "verifiability": 20,
    "form": 15,
    "usage": 25,
    "recency": 10,
}

TIER_LABEL = {
    "high": "核心",
    "good": "支撑",
    "weak": "边缘",
    "drop": "存疑",
}

# 预印本平台（未同行评议）
_PREPRINT_HINTS = (
    "arxiv",
    "biorxiv",
    "medrxiv",
    "ssrn",
    "research square",
    "preprint",
    "techrxiv",
    "chemrxiv",
    "osf",
)
# 同行评议线索（期刊 / 会议 / 出版社）
_PEER_HINTS = (
    "journal",
    "transactions",
    "proceedings",
    "conference",
    "symposium",
    "workshop",
    "letters",
    "nature",
    "science",
    "cell",
    "ieee",
    "acm",
    "springer",
    "elsevier",
    "wiley",
    "vol.",
    "pp.",
    "no.",
)

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\"'）)]+", re.I)
_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
# 编号参考文献行：[1] / 1. / 1)
_NUM_LINE_RE = re.compile(r"^\s*\[?(\d{1,3})[\]\.\)]\s+")


def _present(value: Any) -> bool:
    """判断字段是否"有实质内容"（避免 None / '' / 'unknown' 被当成有效值）。"""
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"none", "null", "unknown", "n/a", "-", "待补"}


def _norm_title(text: str) -> str:
    """标题归一化：去标点、压缩空白、小写，用于重复检测与正文匹配。"""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", (text or "").lower())


def _extract_year(text: str) -> str:
    """从任意文本里取第一个四位年份（1900-2099）。"""
    m = _YEAR_RE.search(text or "")
    return m.group(1) if m else ""


def _extract_doi(text: str) -> str:
    m = _DOI_RE.search(text or "")
    return m.group(0).rstrip(".") if m else ""


def _extract_arxiv(text: str) -> str:
    m = _ARXIV_RE.search(text or "")
    return (m.group(1) + (m.group(2) or "")) if m else ""


def _venue_kind(venue: str) -> str:
    """发表形态：peer（同行评议）/ preprint（预印本）/ unknown。"""
    v = (venue or "").lower()
    if not v:
        return "unknown"
    if any(k in v for k in _PREPRINT_HINTS):
        return "preprint"
    if any(k in v for k in _PEER_HINTS):
        return "peer"
    # 有 venue 但既非预印本也无期刊线索：按"有正式出处"给中间值
    return "peer" if len(v) >= 4 else "unknown"


def _looks_like_authors(seg: str) -> bool:
    """判断一个句段是否像"作者列表"（而不是题名）。

    典型作者段：`Vaswani A, Shazeer N, Parmar N, et al` / `Devlin J, Chang M W`。
    特征：逗号多、单词平均长度短、含姓名缩写（单字母或 "X."）、常以 et al 结尾。
    """
    s = (seg or "").strip()
    if not s or len(s) > 160:
        return False
    if re.search(r"\bet\s+al\.?$", s, re.I):
        return True
    # 单词作者：`Smith J` / `Smith J.` / `van der Waals J`（无逗号）
    if re.fullmatch(r"(?:[A-Z][A-Za-z\-']+\s+){1,3}[A-Z]\.?", s):
        return True
    tokens = re.findall(r"[A-Za-z\u4e00-\u9fff]+", s)
    if not tokens:
        return False
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    initials = len(re.findall(r"\b[A-Z]\b", s))
    return s.count(",") >= 1 and (avg_len <= 7.5 or initials >= 2)


def parse_reference_text(text: str) -> list[dict[str, Any]]:
    """把用户粘贴的参考文献文本解析为条目列表。

    支持三种常见形态（按优先级探测）：
      1) BibTeX（含 `@article{...}`）
      2) RIS（含 `TY  - `）
      3) 纯文本编号列表（`[1] ...` / `1. ...`），无编号时按行切分
    """
    raw = (text or "").strip()
    if not raw:
        return []
    if raw.lstrip().startswith("@") or "\n@" in raw:
        return bibparse.parse_bibtex(raw)
    if "TY  - " in raw:
        return bibparse.parse_ris(raw)

    blocks: list[str] = []
    cur: list[str] = []
    for line in raw.splitlines():
        if _NUM_LINE_RE.match(line):
            if cur:
                blocks.append(" ".join(cur).strip())
            cur = [line]
        elif line.strip():
            cur.append(line.strip())
        elif cur:
            blocks.append(" ".join(cur).strip())
            cur = []
    if cur:
        blocks.append(" ".join(cur).strip())

    # 没有识别到编号：一行一条（过滤过短行，通常是表头/空行）
    if len(blocks) <= 1:
        lines = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) >= 20]
        blocks = lines or blocks

    entries: list[dict[str, Any]] = []
    for block in blocks:
        body = _NUM_LINE_RE.sub("", block).strip()
        if not body:
            continue
        doi = _extract_doi(body)
        arxiv_id = _extract_arxiv(body)
        year = _extract_year(body)

        # 标题 / 作者：引号内标题优先；否则按句点切段，首段若像"作者列表"则次段为标题。
        # 标准参考文献（GB/T 7714、IEEE、ACM）几乎都是「作者. 题名. 出处, 年.」结构。
        quoted = re.search(r"[\"“](.{8,200}?)[\"”]", body)
        segs = [s.strip() for s in re.split(r"\.\s+", body) if s.strip()]
        author_seg = ""
        if quoted:
            title = quoted.group(1)
            author_seg = body.split(quoted.group(0))[0].strip(" .;,")
        else:
            title = segs[0] if segs else body
            head = (segs[0] or "").strip().lower() if segs else ""
            if len(segs) >= 2 and (head.startswith("anonymous") or _looks_like_authors(segs[0])):
                author_seg, title = segs[0], segs[1]
        title = re.sub(r"^\s*\[?\d{1,3}[\]\.\)]\s*", "", title).strip(" .;,")

        if author_seg and not author_seg.strip().lower().startswith("anonymous"):
            authors = [
                a.strip() for a in re.split(r"[,;]", author_seg) if 2 <= len(a.strip()) <= 40
            ][:6]
        else:
            # 匿名 / 无独立作者段：宁可判"缺作者"，也不要把题名当作者
            authors = []
        venue = ""
        m = re.search(r"(?:In|Proceedings of|Journal of)\s+([^.,;]{3,80})", body, re.I)
        if m:
            venue = m.group(1).strip()
        if not venue:
            # 兜底：直接识别常见会议 / 期刊 / 预印本平台名（参考文献常省略 "In"）
            m2 = re.search(
                r"\b(arXiv(?:\s+preprint)?|bioRxiv|medRxiv|NeurIPS|NIPS|CVPR|ICCV|ECCV|ICLR|ICML|"
                r"ACL|NAACL|EMNLP|AAAI|IJCAI|SIGIR|KDD|WWW|IEEE[^.,;]{0,30}|ACM[^.,;]{0,30}|"
                r"Nature|Science|Springer|Elsevier)\b",
                body,
                re.I,
            )
            if m2:
                venue = m2.group(1).strip()
        entries.append(
            {
                "title": title[:300],
                "authors": authors,
                "published": year,
                "venue": venue,
                "doi": doi,
                "url": "",
                "summary": "",
                "pdf_url": (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""),
                "arxiv_id": arxiv_id,
            }
        )
    return entries


def normalize_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """统一 RIS/BibTeX/纯文本/检索结果的字段名，便于统一打分。"""
    out: list[dict[str, Any]] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        title = str(e.get("title") or "").strip()
        authors = e.get("authors") or []
        if isinstance(authors, str):
            authors = [a for a in re.split(r"\s*(?:;|,|\band\b)\s*", authors) if a]
        year = str(e.get("published") or e.get("year") or "")
        year = _extract_year(year) or (year[:4] if year[:4].isdigit() else "")
        doi = str(e.get("doi") or "") or _extract_doi(" ".join(str(v) for v in e.values()))
        arxiv_id = str(e.get("arxiv_id") or "") or _extract_arxiv(
            " ".join(str(v) for v in (e.get("url"), e.get("doi"), e.get("pdf_url"), title))
        )
        out.append(
            {
                "title": title,
                "authors": [str(a).strip() for a in authors if str(a).strip()][:12],
                "year": year,
                "venue": str(e.get("venue") or e.get("journal") or "").strip(),
                "doi": doi,
                "arxiv_id": arxiv_id,
                "url": str(e.get("url") or "").strip(),
                "pdf_url": str(e.get("pdf_url") or "").strip(),
                "source": str(e.get("source") or "").strip(),
            }
        )
    return out


def _count_in_text_mentions(index: int, title: str, body: str) -> int | None:
    """统计第 index 条（1-based）文献在正文中的被引次数；无正文返回 None。

    两条互补路径：
      1) 编号标记：`[3]`、`[3,5]`、`[3-5]`、`[3–5]` —— 展开区间后统计；
      2) 标题词匹配：正文出现该标题至少 2 个实词（作者-年份式引用常无编号）。
    """
    if not body or not body.strip():
        return None
    text = body
    count = 0
    for marker in re.findall(r"\[[0-9,\s\-–—]{1,40}\]", text):
        nums: set[int] = set()
        for part in re.split(r"[,\s]+", marker.strip("[]")):
            if not part:
                continue
            rng = re.match(r"^(\d{1,3})[-–—](\d{1,3})$", part)
            if rng:
                a, b = int(rng.group(1)), int(rng.group(2))
                if a <= b and b - a <= 30:
                    nums.update(range(a, b + 1))
            elif part.isdigit():
                nums.add(int(part))
        if index in nums:
            count += 1

    words = [w for w in re.findall(r"[A-Za-z]{4,}", title or "")]
    words = list(dict.fromkeys(w.lower() for w in words))[:8]
    if words:
        low = text.lower()
        hits = sum(1 for w in words if w in low)
        if hits >= 2:
            count = max(count, 1)
    return count


def score_entry(
    item: dict[str, Any],
    body_text: str = "",
    index: int = 1,
    now_year: int | None = None,
    duplicate: bool = False,
) -> dict[str, Any]:
    """对单条文献打分，返回含逐项信号与中文理由的结果字典。"""
    import datetime

    now_year = now_year or datetime.date.today().year
    title = item.get("title") or ""
    authors = item.get("authors") or []
    year = str(item.get("year") or "")
    venue = item.get("venue") or ""
    doi = item.get("doi") or ""
    arxiv_id = item.get("arxiv_id") or ""
    url = item.get("url") or ""
    pdf_url = item.get("pdf_url") or ""

    reasons: list[str] = []
    flags: list[str] = []

    # ---- A. 元数据完整度（25）----
    meta = 0
    if authors:
        meta += 8 if len(authors) >= 3 else 6
        reasons.append(f"作者信息完整（{len(authors)} 位）")
    else:
        flags.append("missing_author")
        reasons.append("缺少作者信息，无法判断来源权威性")
    if _present(year) and 1900 <= int(year) <= now_year + 1:
        meta += 6
        reasons.append(f"出版年份可识别（{year}）")
    else:
        flags.append("missing_year")
        reasons.append("缺少或无法识别出版年份")
    if _present(venue):
        meta += 5
        reasons.append(f"有发表出处（{venue[:40]}）")
    else:
        flags.append("missing_venue")
    if _present(doi) or _present(arxiv_id):
        meta += 6
    else:
        flags.append("missing_identifier")
        reasons.append("无 DOI / arXiv 标识，难以核验真伪")

    # ---- B. 可核验性（20）----
    verify = 0
    if _present(doi):
        if _DOI_RE.fullmatch(doi.strip()):
            verify += 10
            reasons.append("DOI 格式合法，可在线核验")
        else:
            verify += 3
            flags.append("doi_malformed")
            reasons.append("DOI 格式异常（疑似手写或拼接错误）")
    if _present(arxiv_id):
        verify += 5
    if _present(url) or _present(pdf_url):
        u = (url or pdf_url).lower()
        if any(
            k in u
            for k in (
                "doi.org",
                "arxiv.org",
                "pubmed",
                "ieee",
                "acm.org",
                "springer",
                "sciencedirect",
                "nature.com",
            )
        ):
            verify += 5
            reasons.append("链接指向可信学术站点")
        else:
            verify += 2

    # ---- C. 发表形态（15）----
    kind = _venue_kind(venue)
    if kind == "peer":
        form = 15
        reasons.append("期刊 / 会议正式发表（经同行评议）")
    elif kind == "preprint":
        form = 9
        flags.append("preprint_only")
        reasons.append("仅为预印本，未经同行评议")
    else:
        form = 3
        flags.append("unknown_venue")
        reasons.append("无明确发表形态，引用价值存疑")

    # ---- D. 正文使用强度（25）----
    mentions = _count_in_text_mentions(index, title, body_text)
    if mentions is None:
        usage = 12
        reasons.append("未提供正文，使用强度按中性计分")
    elif mentions >= 3:
        usage = 25
        reasons.append(f"正文实质引用 {mentions} 次，属于关键支撑文献")
    elif mentions == 2:
        usage = 20
        reasons.append("正文引用 2 次，使用较充分")
    elif mentions == 1:
        usage = 14
        reasons.append("正文仅引用 1 次，需结合上下文判断是否礼节性引用")
    else:
        usage = 0
        flags.append("not_cited_in_text")
        reasons.append("正文中未检索到引用标记，疑似凑数条目")

    # ---- E. 时效性（10）----
    recency = 0
    if year.isdigit():
        age = now_year - int(year)
        if age < 0:
            recency = 0
            flags.append("year_in_future")
            reasons.append("出版年份晚于当前年份，数据可疑")
        elif age <= 3:
            recency = 10
            reasons.append("近 3 年发表，时效性强")
        elif age <= 10:
            recency = 7
        elif age <= 20:
            recency = 3
            reasons.append("发表超过 10 年，需确认是否仍为必要引用")
    else:
        recency = 0

    # ---- 扣分项 ----
    penalty = 0
    if duplicate:
        penalty -= 30
        flags.append("duplicate")
        reasons.append("与前面的条目重复（同一文献重复计数）")
    if not _present(title):
        penalty -= 20
        flags.append("missing_title")
        reasons.append("缺少题名，无法定位文献")
    elif len(_norm_title(title)) < 12:
        penalty -= 12
        flags.append("title_too_short")
        reasons.append("题名过短，信息量不足（可能是占位条目）")
    if (
        re.search(r"\bet\s+al\.?\s*$", (title or "").strip(), re.I)
        or "unknown" in (title or "").lower()
    ):
        penalty -= 15
        flags.append("placeholder_title")
        reasons.append("题名含 et al. / unknown 占位符")

    raw = meta + verify + form + usage + recency + penalty
    score = max(0, min(100, int(round(raw))))
    if score >= 70:
        tier = "high"
    elif score >= 50:
        tier = "good"
    elif score >= 30:
        tier = "weak"
    else:
        tier = "drop"

    return {
        **item,
        "index": index,
        "score": score,
        "tier": tier,
        "tier_label": TIER_LABEL[tier],
        "signals": {
            "metadata": meta,
            "verifiability": verify,
            "form": form,
            "usage": usage,
            "recency": recency,
            "penalty": penalty,
        },
        "in_text_mentions": mentions,
        "reasons": reasons[:8],
        "flags": flags,
    }


def assess(
    entries: list[dict[str, Any]],
    body_text: str = "",
    threshold: int = 50,
    now_year: int | None = None,
) -> dict[str, Any]:
    """批量评估：返回 {summary, items}，items 中 keep 表示是否建议保留。"""
    items = normalize_entries(entries)
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        key = _norm_title(item.get("title", ""))
        dup = bool(key) and key in seen
        if key and not dup:
            seen.add(key)
        result = score_entry(item, body_text=body_text, index=i, now_year=now_year, duplicate=dup)
        result["keep"] = result["score"] >= threshold and not dup
        results.append(result)

    kept = [r for r in results if r["keep"]]
    tiers = {t: sum(1 for r in results if r["tier"] == t) for t in TIER_LABEL}
    summary = {
        "total": len(results),
        "kept": len(kept),
        "dropped": len(results) - len(kept),
        "threshold": threshold,
        "avg_score": int(round(sum(r["score"] for r in results) / len(results))) if results else 0,
        "avg_kept_score": int(round(sum(r["score"] for r in kept) / len(kept))) if kept else 0,
        "tiers": tiers,
        "has_body": bool((body_text or "").strip()),
    }
    return {"summary": summary, "items": results}


if __name__ == "__main__":
    # 本地自测：3 条高质量 + 2 条应付式（重复 / 无出处）
    demo = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "published": "2017",
            "venue": "NeurIPS",
            "doi": "10.5555/3295222.3295349",
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "published": "2019",
            "venue": "NAACL",
            "doi": "10.18653/v1/N19-1423",
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "authors": ["Kaiming He"],
            "published": "2016",
            "venue": "CVPR",
            "doi": "",
        },
        {
            "title": "Attention Is All You Need",
            "authors": ["A. Vaswani"],
            "published": "2017",
            "venue": "",
            "doi": "",
        },
        {"title": "et al.", "authors": [], "published": "", "venue": "", "doi": ""},
    ]
    body = "Transformer [1] 与 BERT [2] 的对比见 [1,2]；ResNet [3] 提供主干网络。"
    out = assess(demo, body_text=body)
    print(out["summary"])
    for it in out["items"]:
        print(
            it["index"],
            it["score"],
            it["tier_label"],
            "keep" if it["keep"] else "drop",
            it["reasons"][:3],
        )
