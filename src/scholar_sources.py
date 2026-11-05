# scholar_sources.py
# ============================================================
# 多学术来源统一检索与标识符解析（仿 Zotero 文献收集能力）
#   来源：arXiv / PubMed（免费公开 API）+ 用户自定义 API
#   标识符：DOI（Crossref）/ arXiv ID / PubMed ID 自动补全元数据
#
# 统一论文字段：
#   source / id / title / authors / summary / published /
#   pdf_url / doi / venue
# 知网、Web of Science 无免费公开 API，Google Scholar 无官方 API
# 且反爬严格，故未内置；用户可通过"自定义 API"接入自有数据源。
# ============================================================
from __future__ import annotations

import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, timedelta

import arxiv_tool  # 复用其带代理的 _get 与 arXiv 检索

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CROSSREF_API = "https://api.crossref.org/works/"


# ================= arXiv 来源 =================
def search_arxiv_unified(query: str, max_results: int = 8) -> list[dict]:
    # strict=True：多词用 AND，避免 arXiv 宽匹配 + 时间排序返回无关的最新论文
    papers = arxiv_tool.search_arxiv(
        query, max_results=max_results, sort_by="submittedDate", strict=True
    )
    return [
        {
            "source": "arXiv",
            "id": p["arxiv_id"],
            "title": p["title"],
            "authors": p["authors"],
            "summary": p["summary"],
            "published": p["published"],
            "pdf_url": p["pdf_url"],
            "doi": "",
            "venue": "arXiv 预印本",
        }
        for p in papers
    ]


# ================= PubMed 来源 =================
def _pubmed_fetch(ids: list[str]) -> list[dict]:
    """按 PMID 列表抓取完整元数据（含摘要）。"""
    q = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "xml",
        }
    )
    root = ET.fromstring(arxiv_tool._get(f"{PUBMED_EFETCH}?{q}").decode("utf-8", "ignore"))
    out = []
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//PMID", default="")
        title_el = art.find(".//ArticleTitle")
        title = " ".join("".join(title_el.itertext()).split()) if title_el is not None else ""
        abstract = " ".join(
            " ".join("".join(x.itertext()).split()) for x in art.findall(".//Abstract/AbstractText")
        ).strip()
        journal = art.findtext(".//Journal/Title", default="")
        year = art.findtext(".//JournalIssue/PubDate/Year", default="") or (
            art.findtext(".//JournalIssue/PubDate/MedlineDate", default="")[:4]
        )
        authors = []
        for a in art.findall(".//AuthorList/Author"):
            last, init = a.findtext("LastName", default=""), a.findtext("Initials", default="")
            if last:
                authors.append(f"{last} {init}".strip())
        doi = ""
        for eid in art.findall(".//ArticleIdList/ArticleId"):
            if eid.get("IdType") == "doi":
                doi = eid.text or ""
        out.append(
            {
                "source": "PubMed",
                "id": pmid,
                "title": title,
                "authors": authors,
                "summary": abstract,
                "published": year,
                "pdf_url": "",
                "doi": doi,
                "venue": journal,
            }
        )
    return out


def search_pubmed(query: str, max_results: int = 8) -> list[dict]:
    """PubMed 检索（按日期排序）。多数记录无开放 PDF，pdf_url 为空。"""
    q = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "date",
            "retmode": "json",
        }
    )
    data = json.loads(arxiv_tool._get(f"{PUBMED_ESEARCH}?{q}").decode("utf-8", "ignore"))
    ids = data.get("esearchresult", {}).get("idlist", [])
    return _pubmed_fetch(ids) if ids else []


# ================= 用户自定义 API 来源 =================
OPENALEX_API = "https://api.openalex.org/works"

_TITLE_KEYS = ("title", "display_name", "paper_title", "name", "headline")
_DATE_KEYS = (
    "publication_date",
    "published_date",
    "publicationDate",
    "publishedDate",
    "datePublished",
    "published",
    "date",
    "updated_date",
    "created_date",
    "from_publication_date",
    "publication_year",
    "year",
    "pubdate",
)
_DOI_KEYS = ("doi", "DOI")
_EXTERNAL_ID_KEYS = ("externalIds", "external_ids", "externalids")
_VENUE_KEYS = ("venue", "journal", "container_title", "source", "publisher")
_PAPER_LIST_KEYS = (
    "results",
    "items",
    "data",
    "papers",
    "works",
    "docs",
    "records",
    "entries",
    "hits",
    "rows",
    "result",
)
# 部分站点把论文包在外壳里（DOAJ 的 bibjson、自建服务的 metadata 等）
_WRAPPER_KEYS = ("bibjson", "metadata", "paper", "article", "attributes", "node")


def _s(v) -> str:
    """把任意 JSON 值收敛成短字符串（dict/list 取其中最具代表性的字段）。"""
    if v is None or isinstance(v, bool):
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        for k in ("url", "pdf_url", "pdfUrl", "value", "name", "display_name", "text", "title"):
            got = _s(v.get(k))
            if got:
                return got
        return ""
    if isinstance(v, list):
        for x in v:
            got = _s(x)
            if got:
                return got
    return ""


def _pick(d, keys) -> str:
    """按优先级取第一个非空字段并转成字符串。"""
    if not isinstance(d, dict):
        return ""
    for k in keys:
        if k in d:
            got = _s(d[k])
            if got:
                return got
    return ""


def _has_title(x) -> bool:
    """判断一条记录是否像论文（含标题字段，或外壳里含标题）。"""
    if not isinstance(x, dict):
        return False
    if any(k in x for k in _TITLE_KEYS):
        return True
    for k in _WRAPPER_KEYS:
        inner = x.get(k)
        if isinstance(inner, dict) and any(t in inner for t in _TITLE_KEYS):
            return True
    return False


def _find_paper_list(data, depth: int = 0) -> list:
    """递归寻找 JSON 中最像论文列表的 list[dict]（先试 results/items/data 等常见键）。"""
    if depth > 5:
        return []
    if isinstance(data, list):
        if data and all(isinstance(x, dict) for x in data[:3]):
            if any(_has_title(x) for x in data[:3]):
                return data
        return []
    if isinstance(data, dict):
        for k in _PAPER_LIST_KEYS:
            found = _find_paper_list(data.get(k), depth + 1)
            if found:
                return found
        for v in data.values():
            found = _find_paper_list(v, depth + 1)
            if found:
                return found
    return []


def _flatten_item(it: dict) -> dict:
    """把 {"bibjson": {...}} / {"metadata": {...}} 之类的外壳摊平，便于统一取字段。"""
    if not isinstance(it, dict):
        return {}
    for k in _WRAPPER_KEYS:
        inner = it.get(k)
        if isinstance(inner, dict) and any(t in inner for t in _TITLE_KEYS):
            merged = dict(inner)
            for kk, vv in it.items():
                merged.setdefault(kk, vv)
            return merged
    return it


def _rebuild_inverted(idx) -> str:
    """OpenAlex 的 abstract_inverted_index（词 → 位置列表）还原为摘要文本。"""
    if not isinstance(idx, dict):
        return ""
    slots: dict[int, str] = {}
    for word, spots in idx.items():
        for p in (spots if isinstance(spots, list) else []):
            if isinstance(p, int):
                slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


def _is_pdf_link(u: str) -> bool:
    """粗判一个 URL 是否指向 PDF（避免把落地页当全文链接）。"""
    if not isinstance(u, str):
        return False
    low = u.strip().lower()
    if not low.startswith("http"):
        return False
    return (
        ".pdf" in low
        or "/pdf" in low
        or "pdf?" in low
        or "format=pdf" in low
        or low.endswith("/pdf")
    )


def _deep_pdf(obj, depth: int = 0) -> str:
    """在任意嵌套结构里找 PDF 直链。

    覆盖 OpenAlex（best_oa_location.pdf_url / open_access.oa_url）、
    Semantic Scholar（openAccessPdf.url）、Crossref（link[] + content-type）、
    DOAJ（link[] + type=fulltext）等常见布局。
    """
    if depth > 4:
        return ""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            # pdf_url / openAccessPdf.url 这类"字段名自带 pdf 语义"的直接采用
            if isinstance(v, str) and v.lower().startswith("http"):
                if "pdf" in kl or (
                    kl in ("oa_url", "fulltext_url", "download_url") and _is_pdf_link(v)
                ):
                    return v.strip()
            elif isinstance(v, (dict, list)) and "pdf" in kl:
                cand = _s(v)
                if cand.lower().startswith("http"):
                    return cand
        for k in (
            "best_oa_location",
            "primary_location",
            "open_access",
            "openAccessPdf",
            "oa_location",
            "locations",
            "fulltext",
            "links",
            "link",
            "best_oa",
        ):
            if k in obj:
                got = _deep_pdf(obj[k], depth + 1)
                if got:
                    return got
        for v in obj.values():
            if isinstance(v, (dict, list)):
                got = _deep_pdf(v, depth + 1)
                if got:
                    return got
    elif isinstance(obj, list):
        for it in obj:
            if not isinstance(it, dict):
                continue
            ct = str(it.get("content-type") or it.get("content_type") or "").lower()
            tp = str(it.get("type") or "").lower()
            u = str(it.get("URL") or it.get("url") or "")
            if ("pdf" in ct or "pdf" in tp or "fulltext" in tp) and _is_pdf_link(u):
                return u
        for it in obj:
            got = _deep_pdf(it, depth + 1)
            if got:
                return got
    return ""


def _norm_doi(v: str) -> str:
    """规范化 DOI：去掉 https://doi.org/ 前缀，非 DOI 形态返回空串。"""
    v = (v or "").strip()
    v = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", v, flags=re.I)
    return v if re.match(r"^10\.\d{4,9}/", v) else ""


def _authors(it: dict) -> list[str]:
    """提取作者名单（兼容字符串数组 / {name} / OpenAlex authorships）。"""
    raw = it.get("authors") or it.get("author") or it.get("creators") or it.get("authorships") or []
    if isinstance(raw, dict):
        raw = [raw]
    names: list[str] = []
    for a in (raw if isinstance(raw, list) else []):
        if isinstance(a, dict):
            n = (
                _s(a.get("name"))
                or _s(a.get("display_name"))
                or _s(a.get("author"))
                or f"{_s(a.get('given'))} {_s(a.get('family'))}".strip()
            )
        else:
            n = _s(a)
        if n:
            names.append(n)
    return names[:6]


def _abstract(it: dict) -> str:
    txt = _pick(it, ("abstract", "summary", "description", "abstract_text", "tldr"))
    if not txt:
        txt = _rebuild_inverted(it.get("abstract_inverted_index"))
    return re.sub(r"<[^>]+>", " ", txt).strip()[:1200]


def _paper_date(it: dict) -> str:
    """提取发布时间（兼容字符串 / Crossref 的 date-parts）。"""
    for k in ("published", "issued", "published-print", "published-online", "publication_date"):
        raw = it.get(k)
        if isinstance(raw, dict):
            parts = raw.get("date-parts") or []
            if parts and isinstance(parts[0], list) and parts[0]:
                vals = [str(x) for x in parts[0]]
                # 补零成 YYYY-MM-DD，保证按字符串倒序即按时间倒序
                vals = vals[:1] + [v.zfill(2) for v in vals[1:]]
                return "-".join(vals)[:10]
    return _pick(it, _DATE_KEYS)[:10]


def _paper_doi(it: dict) -> str:
    """提取 DOI：先看 doi 字段，再兼容 Semantic Scholar 的 externalIds.DOI。"""
    d = _norm_doi(_pick(it, _DOI_KEYS))
    if d:
        return d
    for k in _EXTERNAL_ID_KEYS:
        inner = it.get(k)
        if isinstance(inner, dict):
            d = _norm_doi(_s(inner.get("DOI") or inner.get("doi")))
            if d:
                return d
    return ""


def _paper_pdf(it: dict) -> str:
    got = _deep_pdf(it)
    if got:
        return got
    for k in ("url", "link", "href", "landing_page_url", "permalink"):
        u = _s(it.get(k))
        if _is_pdf_link(u):
            return u
    return ""


def _title_sim(a: str, b: str) -> float:
    """标题词集重合度，用于判断兜底检索命中的是不是同一篇。"""
    aw = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    bw = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    return len(aw & bw) / len(aw) if aw else 0.0


def _oa_pdf_by_doi(doi: str) -> str:
    """按 DOI 查 OpenAlex，取开放获取版本（best_oa_location / oa_url）。"""
    if not doi:
        return ""
    try:
        data = json.loads(
            arxiv_tool._get(
                f"{OPENALEX_API}/https://doi.org/{urllib.parse.quote(doi, safe='/')}", timeout=10
            ).decode("utf-8", "ignore")
        )
    except Exception:
        return ""
    return _deep_pdf(data)


def _oa_pdf_by_title(title: str) -> str:
    """按标题查 OpenAlex，标题高度吻合时返回开放获取 PDF。"""
    if not title:
        return ""
    try:
        q = urllib.parse.urlencode({"search": title, "per-page": 1})
        data = json.loads(
            arxiv_tool._get(f"{OPENALEX_API}?{q}", timeout=10).decode("utf-8", "ignore")
        )
    except Exception:
        return ""
    items = _find_paper_list(data)
    if not items:
        return ""
    top = items[0]
    if _title_sim(title, _pick(top, _TITLE_KEYS)) < 0.78:
        return ""
    return _deep_pdf(top)


def _arxiv_pdf_by_title(title: str) -> str:
    """按标题到 arXiv 找预印本版本（期刊论文多为闭源，预印本常有开放 PDF）。"""
    if not title:
        return ""
    try:
        hits = arxiv_tool.search_arxiv(f'"{title}"', max_results=1)
    except Exception:
        return ""
    if not hits:
        return ""
    if _title_sim(title, hits[0].get("title", "")) < 0.8:
        return ""
    return hits[0].get("pdf_url", "")


def _resolve_pdf(p: dict, skip_openalex: bool = False) -> str:
    """PDF 兜底链：OpenAlex(DOI) -> OpenAlex(标题) -> arXiv(标题)。"""
    if not skip_openalex:
        got = (_oa_pdf_by_doi(p.get("doi", "")) if p.get("doi") else "") or _oa_pdf_by_title(
            p.get("title", "")
        )
        if got:
            return got
    return _arxiv_pdf_by_title(p.get("title", ""))


def _fill_missing_pdfs(papers: list[dict], oa_native: set, budget: int = 3, timeout: float = 8.0):
    """并发补齐缺失的 PDF 直链。

    单个站点的兜底查询可能要数秒，串行会把「最新论文」拖到几十秒，
    因此并发执行并设总时长上限：超时未返回的条目保持「无开放 PDF」，不阻塞抓取。
    """
    targets = [(i, p) for i, p in enumerate(papers) if not p.get("pdf_url")][:budget]
    if not targets:
        return
    ex = ThreadPoolExecutor(max_workers=len(targets))
    try:
        futs = {ex.submit(_resolve_pdf, p, i in oa_native): p for i, p in targets}
        done, _ = wait(futs, timeout=timeout)
        for f in done:
            if f.exception() is not None:
                continue
            url = f.result()
            if url:
                futs[f]["pdf_url"] = url
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # Python < 3.9
            ex.shutdown(wait=False)


def _fill_url(template: str, query: str, max_results: int, days: int, api_key: str) -> str:
    """填充 URL 占位符；模板里没写 {query} 时自动追加 query 参数，保证关键词一定带上。

    占位符：{query} 关键词 / {limit}{max_results}{per_page} 条数 /
            {from}{to} 时间窗起止（YYYY-MM-DD）/ {days} 窗口天数 /
            {year} 当前年份 / {api_key} 查询参数式 Key
    """
    today = date.today()
    days = max(0, int(days or 0))
    start = today - timedelta(days=days) if days > 0 else date(1900, 1, 1)
    q = urllib.parse.quote(query)
    repl = {
        "{query}": q,
        "{limit}": str(max_results),
        "{max_results}": str(max_results),
        "{per_page}": str(max_results),
        "{days}": str(days or 3650),
        "{from}": start.isoformat(),
        "{to}": today.isoformat(),
        "{year}": str(today.year),
        "{api_key}": urllib.parse.quote(api_key or ""),
    }
    url = template
    for k, v in repl.items():
        url = url.replace(k, v)
    if (
        "{query}" not in template
        and "query=" not in url.lower()
        and "search=" not in url.lower()
        and "term=" not in url.lower()
    ):
        url += ("&" if "?" in url else "?") + "query=" + q
    # 占位符留空会留下 api_key= 这类空参数，部分接口会直接 401，这里统一剔除
    if "?" in url:
        base, qs = url.split("?", 1)
        kept = [p for p in qs.split("&") if p and not p.endswith("=")]
        url = base + ("?" + "&".join(kept) if kept else "")
    return url


def search_custom(
    api_url: str,
    query: str,
    max_results: int = 8,
    api_key: str = "",
    days: int = 0,
    source_name: str = "",
) -> list[dict]:
    """用户自定义数据源：按研究方向检索「最新」论文，并尽力给出可下载的 PDF 直链。

    - 占位符见 _fill_url；{from}/{to} 配合 days 实现「只看近 N 天」；
    - 鉴权：URL 里写了 {api_key} 就走查询参数，否则同时发送
      Authorization: Bearer 与 x-api-key（兼容 OpenAlex / Semantic Scholar / CORE 等约定）；
    - 字段解析为启发式：深度查找论文列表与 PDF 链接，OpenAlex 倒排摘要自动还原；
    - 无 PDF 时按 DOI / 标题到 OpenAlex 兜底查开放版本（最多 3 条，失败静默）；
    - 返回值按发布时间倒序，保证「最新」在前。
    """
    url = _fill_url(api_url, query, max_results, days, api_key)
    headers = None
    if api_key and "{api_key}" not in api_url:
        headers = {"Authorization": f"Bearer {api_key}", "x-api-key": api_key}
    raw = arxiv_tool._get(url, timeout=25, headers=headers).decode("utf-8", "ignore")
    try:
        data = json.loads(raw)
    except Exception:
        head = re.sub(r"\s+", " ", raw)[:80]
        raise ValueError(f"返回内容不是 JSON，请确认填的是 API 地址而不是文档页/首页：{head}")

    out: list[dict] = []
    oa_native: set[int] = set()  # 本身就是 OpenAlex 记录：开放获取信息已在返回里，无需再查一遍
    for it in _find_paper_list(data)[:max_results]:
        it = _flatten_item(it)
        title = _pick(it, _TITLE_KEYS)
        if not title:
            continue
        doi = _paper_doi(it)
        if "open_access" in it or "best_oa_location" in it:
            oa_native.add(len(out))
        out.append(
            {
                "source": source_name or "自定义API",
                "id": _pick(it, ("id", "paper_id", "uid")) or doi,
                "title": title,
                "authors": _authors(it),
                "summary": _abstract(it),
                "published": _paper_date(it),
                "pdf_url": _paper_pdf(it),
                "doi": doi,
                "venue": _pick(it, _VENUE_KEYS),
            }
        )

    # PDF 兜底：接口只给元数据时，并发布查开放版本 / 预印本（限 3 条、有总时长上限）
    try:
        _fill_missing_pdfs(out, oa_native)
    except Exception:
        pass  # 兜底失败不影响已解析出的元数据

    out.sort(key=lambda p: p["published"] or "0000", reverse=True)
    return out


SOURCES = {
    "arxiv": search_arxiv_unified,
    "pubmed": search_pubmed,
}


def pick_custom_source(custom_sources: list[dict], custom_source_id: str = "") -> dict:
    """在已添加的自定义数据源里按 id 取；id 缺失/失效时退回第一个。"""
    items = [s for s in (custom_sources or []) if isinstance(s, dict) and s.get("url")]
    if not items:
        return {}
    for s in items:
        if s.get("id") == custom_source_id:
            return s
    return items[0]


def pick_custom_sources(
    custom_sources: list[dict],
    custom_source_ids: list[str] | None = None,
    custom_source_id: str = "",
) -> list[dict]:
    """取出本次要抓取的自定义数据源（可多条）。

    - 传了 custom_source_ids：按 id 多选，顺序跟随用户列表；
    - 没传（旧设置/旧前端）：退回单条 custom_source_id（再缺省第一条）。
    """
    items = [s for s in (custom_sources or []) if isinstance(s, dict) and s.get("url")]
    if not items:
        return []
    ids = [str(i) for i in (custom_source_ids or []) if str(i or "").strip()]
    if ids:
        picked = [s for s in items if str(s.get("id") or "") in ids]
        if picked:
            return picked
    one = pick_custom_source(items, custom_source_id)
    return [one] if one else []


def search_sources(
    sources: list[str],
    query: str,
    max_results: int = 8,
    custom_api_url: str = "",
    custom_sources: list[dict] | None = None,
    custom_source_id: str = "",
    custom_source_ids: list[str] | None = None,
) -> tuple[list[dict], str]:
    """按来源列表检索并合并去重；单源失败不影响其他源，错误汇总返回。

    自定义来源：`custom_source_ids` 里的每条都会抓（可同时多个），
    单条失败只记该条的错误；都没有时退回旧字段 custom_api_url（兼容历史设置）。
    """
    merged: list[dict] = []
    errors = []
    for src in sources:
        try:
            if src == "custom":
                specs = pick_custom_sources(
                    custom_sources or [], custom_source_ids, custom_source_id
                )
                if not specs and custom_api_url:
                    specs = [{"url": custom_api_url}]
                for spec in specs:
                    name = str(spec.get("name") or "")
                    try:
                        merged += search_custom(
                            spec.get("url", ""),
                            query,
                            max_results,
                            api_key=spec.get("api_key", ""),
                            days=int(spec.get("days") or 0),
                            source_name=name,
                        )
                    except Exception as e:
                        # 某个自定义源挂了不影响其它源，但要能说清是哪一个
                        errors.append(f"custom{f'[{name}]' if name else ''}: {str(e)[:120]}")
            elif src in SOURCES:
                merged += SOURCES[src](query, max_results)
        except Exception as e:
            errors.append(f"{src}: {str(e)[:120]}")
    # 按标题去重（跨源可能命中同一工作）
    seen, out = set(), []
    for p in merged:
        k = re.sub(r"\W+", "", p["title"].lower())[:60]
        if k and k not in seen:
            seen.add(k)
            out.append(p)
    return out, "; ".join(errors)


# ================= 标识符解析（DOI / arXiv ID / PubMed ID） =================
def resolve_identifier(text: str) -> dict:
    """识别标识符并联网补全元数据，返回统一字段（含 pdf_url 若有开放全文）。"""
    t = text.strip()
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", t)
    if m:
        return _resolve_doi(m.group(1).rstrip(".;,)"))
    m = re.search(r"(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", t, re.I)
    if m:
        return _resolve_arxiv(m.group(1))
    m = re.fullmatch(r"(?:PMID:?\s*)?(\d{6,9})", t, re.I)
    if m:
        return _resolve_pmid(m.group(1))
    raise ValueError("未识别为 DOI / arXiv ID / PubMed ID，请检查输入格式")


def _resolve_arxiv(arxiv_id: str) -> dict:
    p = arxiv_tool.get_arxiv_by_id(arxiv_id)
    if not p:
        raise ValueError(f"arXiv 未找到 ID：{arxiv_id}")
    return {
        "source": "arXiv",
        "id": p["arxiv_id"],
        "title": p["title"],
        "authors": p["authors"],
        "summary": p["summary"],
        "published": p["published"],
        "pdf_url": p["pdf_url"],
        "doi": "",
        "venue": "arXiv 预印本",
    }


def _resolve_pmid(pmid: str) -> dict:
    papers = _pubmed_fetch([pmid])
    if not papers:
        raise ValueError(f"PubMed 未找到 PMID：{pmid}")
    return papers[0]


def _resolve_doi(doi: str) -> dict:
    url = CROSSREF_API + urllib.parse.quote(doi, safe="")
    data = json.loads(arxiv_tool._get(url).decode("utf-8", "ignore"))["message"]
    title = (data.get("title") or [""])[0]
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip() for a in data.get("author", [])
    ][:6]
    issued = (data.get("issued", {}).get("date-parts") or [[""]])[0][0]
    venue = (data.get("container-title") or [""])[0]
    abstract = re.sub(r"<[^>]+>", " ", data.get("abstract", "") or "").strip()
    # Crossref 官方全文链接（常为出版商付费墙）
    pdf_url = _deep_pdf(data.get("link", []))
    # 兜底：OpenAlex 开放版本 -> arXiv 预印本（失败不阻断元数据返回）
    if not pdf_url:
        pdf_url = _oa_pdf_by_doi(doi) or _arxiv_pdf_by_title(title)
    return {
        "source": "Crossref(DOI)",
        "id": doi,
        "title": title,
        "authors": authors,
        "summary": abstract,
        "published": str(issued or ""),
        "pdf_url": pdf_url,
        "doi": doi,
        "venue": venue,
    }
