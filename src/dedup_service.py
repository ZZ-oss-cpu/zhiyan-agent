# dedup_service.py
# ============================================================
# 语义去重与 Human-in-the-Loop 版本检查（对应赛事方案 dedup_service）
#
# 适配说明（相对参考代码的改动）：
#   1. 依赖注入：qdrant / embeddings / collection 由 RAG 模块传入，
#      避免重复打开本地 Qdrant 存储造成文件锁冲突；
#   2. qdrant.search 已弃用，改用 query_points；
#   3. 阈值按 Embedding 类型自适应：在线语义向量 0.88，
#      本地哈希降级向量词面重叠系统性偏低，放宽到 0.75；
#   4. 查重检索排除参考文献块（is_ref），减少引用区噪声误判。
# ============================================================
from __future__ import annotations

ONLINE_THRESHOLD = 0.88  # 在线语义 Embedding 判重阈值（参考代码原值）
# 本地哈希向量降级时的判重阈值：实测精确副本得分 0.71~0.79，
# 同领域不同论文一般 < 0.6，取 0.65 兼顾召回与误报
HASH_FALLBACK_THRESHOLD = 0.65


def check_paper_duplicate(
    rag_mod,
    pdf_title: str,
    pdf_summary_text: str,
    similarity_threshold: float | None = None,
) -> dict:
    """
    在上传解析入库前，计算 标题+摘要 Embedding 相似度，判断是否已存在旧版本。

    rag_mod: source_tracing.py 加载后的模块对象（提供 qdrant / embeddings /
             COLLECTION_NAME / _LocalHashEmbeddings）。
    返回:
      {"is_duplicate": True, "similarity", "existing_paper_id",
       "action_required": "prompt_user_override", "message"}  —— 需人工确认
      {"is_duplicate": False, "action_required": "proceed_upload"} —— 直接入库
    """
    from qdrant_client.models import (  # pyright: ignore[reportMissingImports]
        Filter,
        FieldCondition,
        MatchValue,
    )

    embeddings = rag_mod.embeddings
    if similarity_threshold is None:
        # 本地哈希向量无语义泛化，同论文相似度系统性偏低，阈值相应放宽
        is_hash = isinstance(embeddings, rag_mod._LocalHashEmbeddings)
        default_thr = HASH_FALLBACK_THRESHOLD if is_hash else ONLINE_THRESHOLD
        # 允许用 DEDUP_SIM_THRESHOLD 覆盖（换模型后想微调判重松紧时不必改代码）
        try:
            import config

            similarity_threshold = float(config.DEDUP_SIM_THRESHOLD) or default_thr
        except Exception:
            similarity_threshold = default_thr

    query_vector = embeddings.embed_query(f"{pdf_title}\n{pdf_summary_text}")

    # 在向量库中检索最相似的已存在论文文本块（排除参考文献区）
    hits = rag_mod.qdrant.query_points(
        collection_name=rag_mod.COLLECTION_NAME,
        query=query_vector,
        limit=6,
        query_filter=Filter(must_not=[FieldCondition(key="is_ref", match=MatchValue(value=True))]),
    ).points

    # 聚合 top-3 候选论文（每篇取最高相似度块）
    by_paper: dict = {}
    for h in hits:
        pid = h.payload.get("paper_id")
        if pid and (pid not in by_paper or h.score > by_paper[pid]["score"]):
            by_paper[pid] = {
                "paper_id": pid,
                "score": round(h.score, 4),
                "version": h.payload.get("version", "v1"),
            }
    candidates = sorted(by_paper.values(), key=lambda x: -x["score"])

    if candidates and candidates[0]["score"] >= similarity_threshold:
        top = candidates[0]
        return {
            "is_duplicate": True,
            "similarity": top["score"],
            "existing_paper_id": top["paper_id"],
            "existing_version": top["version"],
            "candidates": candidates[:3],
            "action_required": "prompt_user_override",
            "message": (
                f"检测到与已有论文 [{top['paper_id']}] "
                f"相似度达 {top['score']:.2f}，是否作为新版本替换/跳过？"
            ),
        }

    return {
        "is_duplicate": False,
        "action_required": "proceed_upload",
        "candidates": candidates[:3],
    }
