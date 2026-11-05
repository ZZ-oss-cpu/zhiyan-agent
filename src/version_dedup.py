# version_dedup.py
# ============================================================
# 论文语义版本去重与演进对比模块
#
# 对应 PDF 方案一演示场景：arXiv v1 vs v7 的语义去重与 Diff 高亮。
# 功能：
#   1. compute_semantic_similarity: 计算两篇文章正文的语义相似度
#      （优先使用 Embedding 余弦相似度；无 API Key 时降级为 difflib 文本相似度）
#   2. is_duplicate_version: 相似度超过阈值即判定为同一论文的不同版本
#   3. diff_versions: 对两个版本生成逐行 Diff，高亮结构性修改
# ============================================================
import math
from difflib import SequenceMatcher, unified_diff


def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    """计算两个 Embedding 向量的余弦相似度。"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_semantic_similarity(text_a: str, text_b: str, embed_fn=None) -> float:
    """
    计算两段文本的语义相似度（0~1）。

    embed_fn: 可选，Embedding 函数（如 OpenAIEmbeddings.embed_query），
              传入后使用向量余弦相似度（语义级，抗措辞改写）；
              不传则降级为 difflib 字符级相似度（离线兜底，精度较低）。
    """
    if not text_a or not text_b:
        return 0.0
    if embed_fn is not None:
        return _cosine_similarity(embed_fn(text_a), embed_fn(text_b))
    # 离线兜底：基于字符匹配率的近似相似度
    return SequenceMatcher(None, text_a, text_b).ratio()


def is_duplicate_version(text_a: str, text_b: str, embed_fn=None, threshold: float = 0.85) -> dict:
    """
    判断两篇文章是否为同一论文的不同版本（如 arXiv v1 与 v7）。

    threshold: 判定阈值，默认 0.85（PDF 方案中语义去重的常用经验值）。
    返回: {"is_duplicate": bool, "similarity": float, "threshold": float}
    """
    sim = compute_semantic_similarity(text_a, text_b, embed_fn)
    return {
        "is_duplicate": sim >= threshold,
        "similarity": round(sim, 4),
        "threshold": threshold,
    }


def diff_versions(text_a: str, text_b: str, from_name: str = "v1", to_name: str = "v7") -> str:
    """
    生成两个版本的逐行 unified Diff，用于高亮结构性修改
    （对应演示视频中「v1 -> v7 修订单元高亮」环节）。

    返回 unified diff 格式的字符串；无差异时返回提示文本。
    """
    diff_lines = list(
        unified_diff(
            text_a.splitlines(),
            text_b.splitlines(),
            fromfile=from_name,
            tofile=to_name,
            lineterm="",
        )
    )
    if not diff_lines:
        return f"{from_name} 与 {to_name} 内容完全一致，无修改。"
    return "\n".join(diff_lines)


# ============================================================
# 演示：模拟 arXiv 1706.03762 的 v1 与 v7 版本对比
# ============================================================
if __name__ == "__main__":
    # 模拟同一论文两个版本的正文（v7 相对 v1 有修订和增补）
    paper_v1 = (
        "Attention Is All You Need. We propose the Transformer, "
        "a model architecture relying entirely on attention mechanisms. "
        "Experiments on two machine translation tasks show the model is superior."
    )
    paper_v7 = (
        "Attention Is All You Need. We propose the Transformer, "
        "a model architecture relying entirely on attention mechanisms. "
        "Experiments on two machine translation tasks show the model is superior "
        "in quality while being more parallelizable and requiring less training time. "
        "We also show the Transformer generalizes well to other tasks."
    )
    other_paper = (
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. "
        "We introduce a new language representation model based on bidirectional pre-training."
    )

    # ---- 用例 1：同一论文 v1 vs v7，应判定为重复版本 ----
    # 注：离线 difflib 兜底是字符级匹配，相似度系统性低于语义 Embedding，
    #     离线演示阈值放宽到 0.7；接入真实 Embedding 后建议使用默认 0.85。
    r1 = is_duplicate_version(paper_v1, paper_v7, threshold=0.7)
    print(f"v1 vs v7 相似度: {r1['similarity']} -> 重复版本: {r1['is_duplicate']}")

    # ---- 用例 2：不同论文对比，不应判定为重复 ----
    r2 = is_duplicate_version(paper_v1, other_paper)
    print(f"v1 vs BERT 相似度: {r2['similarity']} -> 重复版本: {r2['is_duplicate']}")

    # ---- 用例 3：输出版本间 Diff，高亮修订单元 ----
    print("\nv1 -> v7 修订单元 Diff:")
    print(diff_versions(paper_v1, paper_v7))
