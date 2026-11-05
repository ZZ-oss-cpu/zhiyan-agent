# bibparse.py
# ============================================================
# 文献列表批量解析（仿 Zotero 批量导入）：BibTeX / RIS
# 输出统一条目字段：
#   title / authors / published / venue / doi / url / summary / pdf_url
# 其中 pdf_url 仅从条目中可确定性的 arXiv 链接推导（eprint 字段、
# url 中的 arxiv.org/abs/ 链接），其余条目由前端提示"无开放 PDF"。
# ============================================================
from __future__ import annotations

import re


def _arxiv_pdf_from(text: str) -> str:
    """从任意文本中提取 arXiv ID 并构造 PDF 链接。"""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", text, re.I)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    m = re.fullmatch(r"(\d{4}\.\d{4,5})(v\d+)?", text.strip())
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    return ""


def parse_bib_file(filename: str, text: str) -> list[dict]:
    """按扩展名或内容特征自动判别 BibTeX / RIS 并解析。"""
    if filename.lower().endswith(".ris") or "\nTY  -" in text:
        return parse_ris(text)
    return parse_bibtex(text)


def parse_bibtex(text: str) -> list[dict]:
    """解析 BibTeX：按 @entry 切分，逐字段正则提取（容忍嵌套花括号不完美的情况）。"""
    entries = []
    for chunk in re.split(r"\n\s*@", text):
        chunk = chunk.strip()
        if not chunk:
            continue

        def field(name: str) -> str:
            m = re.search(
                name + r"\s*=\s*\{(.+?)\}\s*,?\s*(\n|$)", chunk, re.S | re.I
            ) or re.search(name + r'\s*=\s*"(.+?)"\s*,?\s*(\n|$)', chunk, re.S | re.I)
            return " ".join(m.group(1).split()) if m else ""

        title = field("title").strip("{}")
        if not title:
            continue
        authors_raw = field("author")
        doi = field("doi")
        url = field("url")
        eprint = field("eprint")
        entries.append(
            {
                "title": title,
                "authors": [a.strip() for a in authors_raw.split(" and ") if a.strip()][:6],
                "published": field("year")[:4],
                "venue": field("journal") or field("booktitle"),
                "doi": doi,
                "url": url,
                "summary": field("abstract"),
                "pdf_url": _arxiv_pdf_from(eprint) or _arxiv_pdf_from(url) or _arxiv_pdf_from(doi),
            }
        )
    return entries


def parse_ris(text: str) -> list[dict]:
    """解析 RIS：TY 开始、ER 结束；标签映射 TI/T1/AU/PY/DO/UR/AB/JO。"""
    entries, cur = [], None

    def flush():
        if cur and cur.get("title"):
            cur["pdf_url"] = _arxiv_pdf_from(cur.get("url", "")) or _arxiv_pdf_from(
                cur.get("doi", "")
            )
            entries.append(cur)

    for line in text.splitlines():
        if len(line) < 6 or line[2:6] != "  - ":
            continue
        tag, value = line[:2], line[6:].strip()
        if tag == "TY":
            cur = {
                "title": "",
                "authors": [],
                "published": "",
                "venue": "",
                "doi": "",
                "url": "",
                "summary": "",
                "pdf_url": "",
            }
        elif tag == "ER":
            flush()
            cur = None
        elif cur is None:
            continue
        elif tag in ("TI", "T1"):
            cur["title"] = value
        elif tag == "AU":
            cur["authors"].append(value)
        elif tag == "PY":
            cur["published"] = value[:4]
        elif tag in ("JO", "JF", "T2"):
            cur["venue"] = value
        elif tag == "DO":
            cur["doi"] = value
        elif tag == "UR":
            cur["url"] = value
        elif tag == "AB":
            cur["summary"] = value
    flush()  # 文件末尾无 ER 的容错
    for e in entries:
        e["authors"] = e["authors"][:6]
    return entries
