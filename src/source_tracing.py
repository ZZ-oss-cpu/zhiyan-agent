# source_tracing.py
# 延迟注解求值：允许 list[str] 等新式泛型在低版本分析目标下通过静态检查
from __future__ import annotations

import re
import uuid
from typing import Any

# pyright 定向抑制：IDE 分析环境解释器未装依赖时误报，运行时依赖真实存在
try:
    import pymupdf as fitz  # pyright: ignore[reportMissingImports]  # PyMuPDF ≥ 1.24.3 推荐写法
except ImportError:  # 老版本回退
    import fitz  # pyright: ignore[reportMissingImports]  # PyMuPDF：提取 PDF 带坐标的文本块
from qdrant_client import QdrantClient  # pyright: ignore[reportMissingImports]
from qdrant_client.models import (  # pyright: ignore[reportMissingImports]
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

# 加载 config.py 中的 API Key / Embedding 配置（须在实例化 Embeddings 之前执行）
import config
from embeddings import LocalHashEmbeddings, build_embeddings, describe

config.apply()

# 兼容 `dedup_service` 里的 isinstance(embeddings, rag_mod._LocalHashEmbeddings) 判断
_LocalHashEmbeddings = LocalHashEmbeddings


# ============================================================
# 1. 初始化向量库与 Embedding
#    顺序很关键：先算模型维度，再决定集合尺寸（1536 → 384 需要重建，见 reindex_embeddings.py）
# ============================================================
qdrant = QdrantClient(path=config.QDRANT_PATH or "./qdrant_db")

embeddings = build_embeddings()
EMBEDDING_INFO = describe(embeddings)
EMBEDDING_DIM = int(getattr(embeddings, "dim", 0) or 0)

COLLECTION_NAME = config.QDRANT_COLLECTION or "academic_papers"


def _collection_dim(name: str) -> int | None:
    """读取集合的向量维度；集合不存在返回 None。"""
    try:
        info = qdrant.get_collection(name)
        vectors = info.config.params.vectors
        # 单向量配置是 VectorParams，多向量配置是 dict
        return int(getattr(vectors, "size", 0) or 0) or None
    except Exception:
        return None


def _recreate_collection(name: str, dim: int):
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


_existing_dim = _collection_dim(COLLECTION_NAME)
if _existing_dim is None:
    _recreate_collection(COLLECTION_NAME, EMBEDDING_DIM)
    print(f"[向量库] 已创建集合 {COLLECTION_NAME}（{EMBEDDING_DIM} 维，{EMBEDDING_INFO['label']}）")
elif _existing_dim != EMBEDDING_DIM:
    # 维度不一致：空库直接按新维度重建；有数据则回退到与该库匹配的哈希向量，
    # 保证服务可用，同时明确提示离线重建命令（重建期间会改写全库向量）
    try:
        point_count = qdrant.count(COLLECTION_NAME, exact=True).count
    except Exception:
        point_count = 0
    if point_count == 0:
        _recreate_collection(COLLECTION_NAME, EMBEDDING_DIM)
        print(f"[向量库] 空集合已按新维度重建（{EMBEDDING_DIM} 维）")
    else:
        print(
            f"[向量库] 维度不匹配：库内 {_existing_dim} 维 / 当前模型 {EMBEDDING_DIM} 维，"
            "已回退哈希向量以保持检索可用。\n"
            "         请停止服务后执行：python reindex_embeddings.py（按当前模型重建全库向量）"
        )
        if _existing_dim == LocalHashEmbeddings.dim:
            embeddings = LocalHashEmbeddings()
            EMBEDDING_INFO = describe(embeddings)
            EMBEDDING_DIM = embeddings.dim
        else:
            # 旧库维度既不是新模型也不是哈希：换用带维度的新集合，避免写入报错
            COLLECTION_NAME = f"{COLLECTION_NAME}_{EMBEDDING_DIM}"
            _recreate_collection(COLLECTION_NAME, EMBEDDING_DIM)
            print(f"[向量库] 已改用新集合 {COLLECTION_NAME}（旧集合数据需重建后可见）")


# ============================================================
# OCR（扫描件）支持：RapidOCR —— PaddleOCR PP-OCR 模型的 ONNX 轻量版
#   选型理由（详见《符合度分析.md》P1.1）：
#   1) 下游只需要「文本+位置框」，RapidOCR 的检测框可直接映射为
#      与 PyMuPDF 同构的块坐标，锚点/去重/高亮链路零改动；
#   2) 只依赖 onnxruntime + opencv，CPU 可跑、离线可用、模型随包分发；
#   3) 未安装时保留原有 422 提示（报错里给出安装命令），不影响原生 PDF 路径。
# ============================================================
_OCR_ENGINE = None
_OCR_DPI = 200  # 渲染分辨率：兼顾小字号识别与 CPU 速度（每页约 2~6 秒）
_OCR_SCORE_MIN = 0.5  # 识别置信度阈值，低于此值的框丢弃
_OCR_MIN_PAGE_CHARS = 20  # 单页原生文本少于该字符数 -> 视为扫描页，走 OCR


def _get_ocr():
    """懒加载 OCR 引擎：优先 rapidocr（v3+），兼容旧包名 rapidocr_onnxruntime。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr import RapidOCR  # pyright: ignore[reportMissingImports]
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR  # pyright: ignore[reportMissingImports]
            except ImportError as e:
                raise RuntimeError(
                    "扫描件需要 OCR 引擎，请先安装：pip install rapidocr "
                    "（或旧包名 pip install rapidocr-onnxruntime）"
                ) from e
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _run_ocr_lines(image_bytes: bytes) -> list[tuple[list, str, float]]:
    """对一张整页图片跑 det+rec，返回 [(四点框, 文本, 置信度)]。

    兼容 rapidocr 两个大版本的返回结构：
    v3 返回 RapidOCROutput（.boxes/.txts/.scores）；v1 返回 (result, elapse) 二元组。
    """
    out = _get_ocr()(image_bytes)
    items: list[tuple[list, str, float]] = []
    if out is None:
        return items
    if hasattr(out, "boxes"):  # rapidocr v3
        if out.boxes is None:
            return items
        txts = getattr(out, "txts", None) or []
        scores = getattr(out, "scores", None) or []
        for box, txt, score in zip(out.boxes, txts, scores):
            if txt is not None and str(txt).strip():
                items.append((box, str(txt), float(score)))
        return items
    if isinstance(out, tuple):  # rapidocr v1
        out = out[0]
    if not out:
        return items
    for box, txt, score in out:
        if txt is not None and str(txt).strip():
            items.append((box, str(txt), float(score)))
    return items


def _order_lines_by_columns(lines: list[dict], page_width: float) -> list[dict]:
    """阅读序排序：检测双栏中缝（x 中心最大间隙）则先左栏后右栏，否则按 y 排。

    学术 PDF 多为双栏，纯 y 排序会把左右栏同一行交错拼接、破坏语义连贯。
    """

    def by_pos(line: dict):
        return (round(line["cy"], 1), line["x0"])

    if len(lines) < 4:
        return sorted(lines, key=by_pos)
    xs = sorted(line["cx"] for line in lines)
    gap, split = max((xs[i + 1] - xs[i], (xs[i] + xs[i + 1]) / 2) for i in range(len(xs) - 1))
    if gap > page_width * 0.12:  # 间隙超过页宽 12% 才认为是栏间留白
        left = [line for line in lines if line["cx"] < split]
        right = [line for line in lines if line["cx"] >= split]
        if len(left) >= 2 and len(right) >= 2:
            return sorted(left, key=by_pos) + sorted(right, key=by_pos)
    return sorted(lines, key=by_pos)


def _merge_lines_to_blocks(lines: list[dict]) -> list[dict]:
    """把 OCR 行合并成段落块：起点相近且垂直间距在约 1.6 倍行高内视为同段。

    对齐 PyMuPDF 的块粒度（段落级），保证 chunk 大小与原生 PDF 路径相当，
    避免逐行入库导致检索单元过碎。
    """
    blocks: list[dict] = []
    for ln in lines:
        if blocks:
            prev = blocks[-1]
            v_gap = ln["y0"] - prev["y1"]
            aligned = abs(ln["x0"] - prev["x0"]) < 40 or ln["x0"] <= prev["x1"] + 20
            if aligned and -prev["h"] * 0.6 < v_gap < prev["h"] * 1.6:
                prev["x0"] = min(prev["x0"], ln["x0"])
                prev["x1"] = max(prev["x1"], ln["x1"])
                prev["y1"] = ln["y1"]
                prev["h"] = max(prev["h"], ln["h"])
                prev["text"] = (prev["text"] + " " + ln["text"]).strip()
                continue
        blocks.append(dict(ln))
    return blocks


def _ocr_page_blocks(page) -> list[tuple]:
    """整页渲染 -> OCR -> 双栏排序 -> 段落聚合。

    返回与 PyMuPDF get_text("blocks") 同构的
    (x0, y0, x1, y1, text, block_no, block_type) 元组列表；
    坐标为 PDF 点、origin 左上（get_pixmap 渲染方向与 PyMuPDF 页面坐标一致，
    只需按 dpi 缩放，无需翻转 y 轴）。
    """
    pix = page.get_pixmap(dpi=_OCR_DPI)
    k = 72.0 / _OCR_DPI  # 像素 -> PDF 点
    lines: list[dict] = []
    for box, txt, score in _run_ocr_lines(pix.tobytes("png")):
        if score < _OCR_SCORE_MIN:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x0, x1 = min(xs) * k, max(xs) * k
        y0, y1 = min(ys) * k, max(ys) * k
        text = " ".join(txt.split())
        if not text:
            continue
        lines.append(
            {
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "text": text,
                "cx": (x0 + x1) / 2,
                "cy": (y0 + y1) / 2,
                "h": max(y1 - y0, 1.0),
            }
        )
    if not lines:
        return []
    ordered = _order_lines_by_columns(lines, page.rect.width)
    return [
        (b["x0"], b["y0"], b["x1"], b["y1"], b["text"], 0, 0)
        for b in _merge_lines_to_blocks(ordered)
    ]


def extract_chunks_from_pdf(pdf_path: str, min_len: int = 30):
    """
    提取带页码(page_num)和坐标(bbox)的文本块，坐标为 PDF 点、origin 左上（PyMuPDF 约定）。

    原生 PDF 走 PyMuPDF 文本层；扫描件（无文本层）自动降级为 RapidOCR：
    渲染整页 -> det+rec -> 行排序（双栏中缝检测）-> 聚合成段落块，
    产出与 PyMuPDF 完全同构的 (x0, y0, x1, y1, text, ...) 元组，
    因此 is_ref 标记 / 向量入库 / 锚点高亮全部零改动复用。
    """
    chunks = []
    with fitz.open(pdf_path) as doc:
        # ---- 第一遍：定位 References 标题位置（页码 + y 坐标）----
        # 双栏排版下 PyMuPDF 块顺序是内容流顺序而非视觉顺序，
        # "先遇到标题再标记"的单遍扫描会漏掉流中排在标题之前的条目块
        ref_page, ref_y = None, None
        for page_num, page in enumerate(doc, start=1):
            for block in page.get_text("blocks"):
                t = " ".join(block[4].split()).strip().lower().rstrip(".")
                if t in ("references", "reference"):
                    ref_page, ref_y = page_num, block[1]
                    break
            if ref_page is not None:
                break

        # ---- 第二遍：提取文本块并按位置标记 is_ref（扫描页自动走 OCR）----
        for page_num, page in enumerate(doc, start=1):
            blocks = list(page.get_text("blocks"))
            if sum(len(" ".join(b[4].split())) for b in blocks) < _OCR_MIN_PAGE_CHARS:
                # 无文本层（扫描件）：OCR 兜底，产出同构块
                blocks = _ocr_page_blocks(page)
            for block in blocks:
                # block: (x0, y0, x1, y1, text, block_no, block_type)
                text = " ".join(block[4].split())
                if len(text) < min_len:
                    continue
                # 扫描件第一遍定位不到 References 标题（无文本层）：第二遍顺手补记
                t_head = text.strip().lower().rstrip(".")
                if ref_page is None and t_head in ("references", "reference"):
                    ref_page, ref_y = page_num, block[1]
                # 位置规则：标题所在页之后的所有块，或同页位于标题下方的块
                is_ref = False
                if ref_page is not None and ref_y is not None:
                    is_ref = page_num > ref_page or (page_num == ref_page and block[1] >= ref_y - 5)
                # 内容兜底：文献条目典型特征（[n] 编号开头 / 出版信息），
                # 覆盖双栏排版中位置规则漏掉的条目（如右栏顶部的续条）
                if not is_ref and re.search(
                    r"^\[\d+\]\s*[A-Z]|In Proceedings of|arXiv preprint|IEEE Transactions",
                    text,
                ):
                    is_ref = True
                chunks.append(
                    {
                        "text": text,
                        "page_num": page_num,
                        "bbox": [round(v, 2) for v in block[:4]],
                        "is_ref": is_ref,
                    }
                )
    return chunks


def ingest_pdf_with_anchors(pdf_path: str, paper_id: str):
    """
    解析 PDF 并将带页码坐标的文本块写入 Qdrant，
    支撑 Source Tracing（点击回答跳转 PDF 原图高亮画框）。
    """
    extracted_chunks = extract_chunks_from_pdf(pdf_path)
    if not extracted_chunks:
        print(f"警告: {pdf_path} 未提取到文本块（可能是扫描件，需走 OCR 分支）")
        return

    # 批量嵌入：远程接口一次请求多段、本地 ONNX 也走 batch，比逐段 embed_query 快一个量级
    texts = [c["text"] for c in extracted_chunks]
    vectors: list[list[float]] = []
    batch = max(1, int(config.EMBEDDING_BATCH_SIZE or 16))
    for start in range(0, len(texts), batch):
        vectors.extend(embeddings.embed_documents(texts[start : start + batch]))

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"paper_id": paper_id, **chunk},
        )
        for chunk, vec in zip(extracted_chunks, vectors)
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(
        f"成功将论文 {paper_id} 的 {len(points)} 个文本块解析并存入向量库"
        f"（{EMBEDDING_INFO['label']}，{EMBEDDING_DIM} 维）！"
    )


def retrieve_with_anchors(query: str, paper_id: str | None = None, top_k: int = 3):
    """
    检索最相关的文本块，返回 text / page_num / bbox / score，
    供 RAG 回答时附带来源锚点。
    """
    # 过滤条件：must_not 排除参考文献条目块（is_ref=True 为噪声源）；
    # 指定论文时叠加 paper_id 过滤。无 is_ref 字段的旧数据不受 must_not 影响
    must_not = [FieldCondition(key="is_ref", match=MatchValue(value=True))]
    if paper_id:
        query_filter = Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))],
            must_not=must_not,
        )
    else:
        query_filter = Filter(must_not=must_not)

    # 候选放大：本地哈希向量的余弦分数天然偏爱短块（标题/表格中命中词
    # 占比高），直接取 top_k 容易漏掉内容密集的正文段落；故先多取候选，
    # 重排/去重后再截取 top_k
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embeddings.embed_query(query),
        limit=max(top_k * 6, 20),
        query_filter=query_filter,
    ).points

    # 仅本地哈希向量需要词面重排（在线 Embedding 的语义分数更可靠，保持原序）：
    # 按"查询词在块文本中的绝对命中数"降序，正文长段落因此优先于标题/表格；
    # 若全部候选零命中（跨语言查询未改写等），退回原向量分数序
    if isinstance(embeddings, _LocalHashEmbeddings) and hits:
        q_tokens = set(_LocalHashEmbeddings._tokenize(query))

        def hit_count(text: str) -> int:
            return len(q_tokens & set(_LocalHashEmbeddings._tokenize(text)))

        # 排序键：命中数优先；并列时偏好长块（正文段落 > 章节短标题）
        ranked = sorted(
            hits,
            key=lambda h: (
                hit_count(h.payload["text"]),
                min(len(h.payload["text"]), 300),
                h.score,
            ),
            reverse=True,
        )
        if hit_count(ranked[0].payload["text"]) > 0:
            hits = ranked

    # 按文本内容去重：同一文本块可能因重复块/重传入库出现多个点位，
    # 不去重会让重复锚点挤占 top_k 名额、拉低上下文多样性
    results, seen = [], set()
    for h in hits:
        if h.payload["text"] in seen:
            continue
        seen.add(h.payload["text"])
        results.append(
            {
                "text": h.payload["text"],
                "page_num": h.payload["page_num"],
                "bbox": h.payload["bbox"],
                "score": h.score,
                # 附带所属论文：多论文遍历检索时前端锚点按各自论文跳转
                "paper_id": h.payload.get("paper_id"),
            }
        )
        if len(results) >= top_k:
            break
    return results


def fetch_paper_head_chunks(paper_id: str, max_chunks: int = 2) -> list[dict[str, Any]]:
    """确定性获取论文首页的非参考文献块（摘要/引言）。

    全文级提问（"核心方法/总结/综述"）时，摘要是概括方法的最关键依据，
    但向量检索不保证稳定命中首页——故绕开相似度，按页码直接取。
    """
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="paper_id", match=MatchValue(value=paper_id)),
                FieldCondition(key="page_num", match=MatchValue(value=1)),
                FieldCondition(key="is_ref", match=MatchValue(value=False)),
            ]
        ),
        limit=max_chunks,
        with_payload=True,
    )
    return [
        {
            "text": p.payload["text"],
            "page_num": p.payload["page_num"],
            "bbox": p.payload["bbox"],
            "score": None,  # 非相似度检索，无相关度分数
            "paper_id": paper_id,
        }
        for p in points
    ]


# 测试运行
if __name__ == "__main__":
    ingest_pdf_with_anchors("1706.03762v7.pdf", paper_id="Attention_v7")
