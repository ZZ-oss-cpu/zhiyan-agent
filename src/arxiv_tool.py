# arxiv_tool.py
# ============================================================
# 外部学术 API 工具（对应评分维度：自主工具调用与闭环探索）
#   1. search_arxiv:    arXiv 公开 API 检索（关键词 / 最新提交排序）
#   2. download_pdf:    按 PDF 链接下载论文到本地
#   3. search_github:   GitHub 复现性查询（最相关仓库 stars / 更新日期）
#
# 网络说明：arXiv / GitHub 均为免 Key 公开接口；如需代理，
# 设置 HTTPS_PROXY 环境变量即可（urllib 自动识别）。
# 所有函数在网络不可达时抛出异常，由调用方降级为友好提示。
# ============================================================
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import config  # 代理配置（HTTPS_PROXY / ALLOW_INSECURE_TLS），仅本模块学术请求定向使用

ARXIV_API = "https://export.arxiv.org/api/query"
GITHUB_API = "https://api.github.com/search/repositories"
_UA = {"User-Agent": "zhiyan-agent/1.0 (academic research assistant)"}

_insecure_warned = False  # 证书降级只提示一次，避免刷屏


def _open(req, proxy: str, timeout: int, insecure: bool = False):
    """构造 opener 并发起请求：proxy 非空走代理；insecure=True 时不校验 TLS 证书。"""
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    if insecure:
        handlers.append(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    opener = urllib.request.build_opener(*handlers) if handlers else urllib.request.build_opener()
    return opener.open(req, timeout=timeout)


def _is_cert_error(err: Exception) -> bool:
    """判断是否为 TLS 证书校验失败（校园网/安全软件替换证书的典型表现）。"""
    reason = getattr(err, "reason", err)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return True
    text = f"{reason}".lower()
    return ("certificate verify failed" in text) or ("certificate_verify_failed" in text)


def _get(url: str, timeout: int = 20, headers: dict | None = None) -> bytes:
    """发起 GET 请求。

    - config.HTTPS_PROXY 非空时仅本模块学术请求走代理（不影响 LLM 内网直连）；
    - headers 用于自定义数据源的鉴权头（如 Authorization: Bearer xxx）；
    - 若因证书链不被信任而失败，且 config.ALLOW_INSECURE_TLS 打开，则自动降级重试一次
      （不少校园网/安全软件会用自己的证书解密 HTTPS，Python 默认不信任该证书；
      学术公开数据场景下该降级是务实取舍，可在 config.py 关掉）。
    """
    merged = dict(_UA)
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, headers=merged)
    proxy = getattr(config, "HTTPS_PROXY", "")
    try:
        return _open(req, proxy, timeout).read()
    except urllib.error.URLError as e:
        if not (_is_cert_error(e) and getattr(config, "ALLOW_INSECURE_TLS", True)):
            raise
        global _insecure_warned
        if not _insecure_warned:
            print(
                "[提示] TLS 证书校验失败（多为校园网/安全软件中间证书），已自动降级为不校验重试。"
            )
            _insecure_warned = True
        return _open(req, proxy, timeout, insecure=True).read()


# AND 组合时要剔除的英文虚词：像 all:of 这种条件会让 arXiv 直接返回 0 条
_QUERY_STOP = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "with",
    "by",
    "from",
    "via",
    "using",
    "into",
    "is",
    "are",
    "be",
    "as",
    "we",
    "our",
}


def _build_search_query(query: str, strict: bool) -> str:
    """构造 arXiv 的 search_query。

    strict=True 时把多词查询写成 `all:a AND all:b`：否则 arXiv 会把
    `all:skeleton action recognition` 当成宽松匹配（实测命中 34 万条），
    再配 submittedDate 排序就会返回大量与主题无关的最新论文。
    拼接前会剔除 of / for / the 这类虚词，避免 `all:of` 把命中数打到 0。
    """
    q = (query or "").strip()
    if not strict or not q:
        return f"all:{q}"
    # 已带字段前缀（all:/ti:/abs:…）时原样交给 arXiv，避免拼成 all:all:xxx
    if re.match(r"^[A-Za-z_]+:", q):
        return q
    # 短语引号 / 布尔运算符同理，只补 all: 前缀
    if '"' in q or " AND " in q.upper() or " OR " in q.upper():
        return f"all:{q}"
    words = [w for w in q.split() if w]
    kept = [w for w in words if w.lower().strip(",.;:") not in _QUERY_STOP]
    if not kept:
        kept = words
    if len(kept) <= 1:
        return f"all:{kept[0]}"
    return " AND ".join(f"all:{w}" for w in kept)


def search_arxiv(
    query: str, max_results: int = 5, sort_by: str = "relevance", strict: bool = False
) -> list[dict]:
    """
    检索 arXiv，返回论文列表：
    [{arxiv_id, base_id, version, title, authors, summary, published, pdf_url}]
    sort_by: "relevance"（相关度）或 "submittedDate"（最新提交，定期抓取用）。
    strict: True 时多词用 AND 组合（"最新论文"面板走这条，避免宽匹配污染结果）。
    """
    params = urllib.parse.urlencode(
        {
            "search_query": _build_search_query(query, strict),
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
    )
    xml_text = _get(f"{ARXIV_API}?{params}").decode("utf-8", "ignore")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)

    papers = []
    for entry in root.findall("a:entry", ns):
        arxiv_url = entry.findtext("a:id", default="", namespaces=ns)
        arxiv_id = arxiv_url.rsplit("/", 1)[-1]  # 如 1706.03762v7
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        title = " ".join(entry.findtext("a:title", default="", namespaces=ns).split())
        summary = " ".join(entry.findtext("a:summary", default="", namespaces=ns).split())
        published = entry.findtext("a:published", default="", namespaces=ns)[:10]
        authors = [
            a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)
        ]
        pdf_url = ""
        for link in entry.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "base_id": base_id,
                "version": arxiv_id[len(base_id) :] or "v1",  # v1 / v7 ...
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published,
                "pdf_url": pdf_url,
            }
        )
    return papers


def get_arxiv_by_id(arxiv_id: str) -> dict | None:
    """按 arXiv ID 精确获取单篇论文元数据（标识符导入用）。"""
    base = re.sub(r"v\d+$", "", arxiv_id.strip())
    params = urllib.parse.urlencode({"id_list": base, "max_results": 1})
    xml_text = _get(f"{ARXIV_API}?{params}").decode("utf-8", "ignore")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    arxiv_url = entry.findtext("a:id", default="", namespaces=ns)
    full_id = arxiv_url.rsplit("/", 1)[-1]
    pdf_url = ""
    for link in entry.findall("a:link", ns):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
    return {
        "arxiv_id": full_id,
        "base_id": base,
        "version": full_id[len(base) :] or "v1",
        "title": " ".join(entry.findtext("a:title", default="", namespaces=ns).split()),
        "authors": [
            a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)
        ],
        "summary": " ".join(entry.findtext("a:summary", default="", namespaces=ns).split()),
        "published": entry.findtext("a:published", default="", namespaces=ns)[:10],
        "pdf_url": pdf_url,
    }


_PDF_LINK_PATTERNS = (
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
    r'<link[^>]+type=["\']application/pdf["\'][^>]+href=["\']([^"\']+)',
    r'<a[^>]+href=["\']([^"\']+\.pdf[^"\']*)["\']',
)


def _find_pdf_in_html(html: str, base_url: str) -> str:
    """从文章落地页里找真正的 PDF 链接（citation_pdf_url 优先）。"""
    for pat in _PDF_LINK_PATTERNS:
        m = re.search(pat, html, re.I)
        if m:
            return urllib.parse.urljoin(base_url, m.group(1))
    return ""


def download_pdf(pdf_url: str, save_path: str, timeout: int = 60) -> str:
    """下载论文 PDF 到本地路径，返回保存路径。

    部分数据源给的是文章落地页（HTML）而非 PDF 直链，
    此时从页面里解析 citation_pdf_url 等标注再取一次，提高「获取 PDF」成功率。
    """
    data = _get(pdf_url, timeout=timeout)
    if not data.startswith(b"%PDF"):
        alt = _find_pdf_in_html(data.decode("utf-8", "ignore")[:200000], pdf_url)
        if alt and alt != pdf_url:
            data = _get(alt, timeout=timeout)
    if not data.startswith(b"%PDF"):
        raise ValueError("下载内容不是有效 PDF（可能被重定向到人机校验页）")
    with open(save_path, "wb") as f:
        f.write(data)
    return save_path


def search_github(title: str, per_page: int = 3) -> dict | None:
    """
    GitHub 复现性查询：按论文标题搜仓库，返回最相关仓库信息
    {repo, url, stars, updated, desc}；无结果返回 None。
    未认证限流 10 次/分钟，调用方需控制频率。
    """
    q = urllib.parse.quote(f'"{title}" in:name,description,readme')
    url = f"{GITHUB_API}?q={q}&sort=stars&order=desc&per_page={per_page}"
    data = json.loads(_get(url).decode("utf-8", "ignore"))
    items = data.get("items") or []
    if not items:
        return None
    top = items[0]
    return {
        "repo": top["full_name"],
        "url": top["html_url"],
        "stars": top["stargazers_count"],
        "updated": (top.get("updated_at") or "")[:10],
        "desc": (top.get("description") or "")[:120],
    }


# 简单本地验证
if __name__ == "__main__":
    try:
        results = search_arxiv("skeleton-based action recognition", max_results=2)
        for p in results:
            print(f"{p['arxiv_id']} | {p['published']} | {p['title'][:60]}")
    except Exception as e:
        print(f"arXiv 不可达（当前网络）: {e}")
