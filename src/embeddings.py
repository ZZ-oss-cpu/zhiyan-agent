# embeddings.py
# ============================================================
# 统一 Embedding 工厂（RAG 检索精度问题的根治点）
#
# 背景：早期版本平台未提供向量模型，`source_tracing.py` 只能用
#       `_LocalHashEmbeddings`（词元哈希 + 词面重叠），语义泛化与跨语言能力都很弱：
#         - 中文提问查英文论文，余弦分数趋近 0（只能靠上游 LLM 先改写英文关键词）；
#         - 同义改写（"few-shot" vs "low-data"）完全无法命中。
#
# 现在提供三种后端，按优先级自动选择并对外暴露统一接口：
#   1. remote —— OpenAI 兼容 /embeddings 接口（阿里云百炼 / 硅基流动 / OpenAI 等）；
#   2. local  —— fastembed（ONNX Runtime，CPU 可跑、无需 torch、离线可用）：
#                默认 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
#                （50+ 语言，中英跨语言检索，384 维，约 240MB）；
#   3. hash   —— 词元哈希兜底（1536 维），保证任何环境流程都能跑通。
#
# 统一接口（三种后端一致）：
#   embed_query(text) -> list[float]
#   embed_documents(texts) -> list[list[float]]
#   属性：backend / model_name / dim / is_semantic
#
# 配置（config.py / .env / 环境变量）：
#   EMBEDDING_BACKEND=auto|remote|local|hash
#   EMBEDDING_MODEL=...            远程模型名（留空则不走远程）
#   EMBEDDING_LOCAL_MODEL=...      本地 ONNX 模型名
#   EMBEDDING_CACHE_DIR=./models   本地模型缓存目录（首次会自动下载）
#   EMBEDDING_ALLOW_DOWNLOAD=true  离线答辩时设为 false，只读本地缓存
#   EMBEDDING_BATCH_SIZE=16        批量嵌入大小
# ============================================================
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import time
from typing import Any, Iterable, Sequence

import config

# 本地默认模型：多语言（中英）语义向量，384 维；CPU 上 ~200 段/秒
DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 需要加前缀的模型族（E5 系列要求 query/passage 前缀才能发挥效果）
_E5_PREFIX = "e5"


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    """L2 归一化：Qdrant 用 COSINE 距离，归一化后内积即余弦相似度。"""
    norm = math.sqrt(sum(float(v) * float(v) for v in vec)) or 1.0
    return [float(v) / norm for v in vec]


def _model_family(model_name: str) -> str:
    name = (model_name or "").lower()
    if _E5_PREFIX in name:
        return "e5"
    return "plain"


class LocalHashEmbeddings:
    """离线哈希 Embedding：词元 -> 1536 维稀疏向量（兜底，绝非语义向量）。

    保留该类有两个作用：
      1. 任何环境（无网 / 无模型缓存）都能跑通入库与检索链路；
      2. `dedup_service` 通过 `isinstance(rag._LocalHashEmbeddings)` 自动放宽判重阈值。
    """

    backend = "hash"
    model_name = "local-hash-1536"
    is_semantic = False
    dim = 1536

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中英混合分词：英文/数字整词，中文单字 + 双字滑窗。"""
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        cjk = re.findall(r"[\u4e00-\u9fff]", text or "")
        tokens.extend(cjk)
        tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
        return tokens

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in self._tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return _l2_normalize(vec)

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class LocalOnnxEmbeddings:
    """本地 ONNX 语义向量（fastembed）：无需 API Key、无需 torch、可完全离线。

    首次使用会从 HuggingFace 下载 ONNX 模型到 `EMBEDDING_CACHE_DIR`（默认 ./models）；
    之后断网也能直接加载（`EMBEDDING_ALLOW_DOWNLOAD=false` 可强制只用缓存）。
    """

    backend = "local"
    is_semantic = True

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 16,
        allow_download: bool = True,
        query_prefix: str = "",
    ):
        try:
            from fastembed import TextEmbedding  # pyright: ignore[reportMissingImports]
        except ImportError as e:  # pragma: no cover - 依赖缺失时的明确提示
            raise RuntimeError(
                "未安装 fastembed，无法启用本地语义 Embedding；"
                "请执行 pip install fastembed（或改用 EMBEDDING_BACKEND=hash）"
            ) from e

        self.model_name = model_name
        self.batch_size = max(1, int(batch_size or 16))
        self._family = _model_family(model_name)
        # 查询指令前缀（Qwen3-Embedding 官方建议写法；MiniLM/BGE 留空即可）
        self.query_prefix = (query_prefix or "").replace("\\n", "\n")

        cache_dir = cache_dir or os.path.join(config.BASE_DIR, "models")
        os.makedirs(cache_dir, exist_ok=True)

        # 关闭自动下载：给 huggingface_hub 置离线标志，模型不在缓存时直接报错
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        if not allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads,
                lazy_load=False,
            )
        finally:
            if not allow_download:
                if old_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_offline

        # 探测维度：一次极短文本推理（比读私有属性稳）
        probe = self._raw_embed(["dimension probe"])[0]
        self.dim = len(probe)

    def _raw_embed(self, texts: Sequence[str], query: bool = False) -> list[list[float]]:
        """调用 fastembed；E5 系列按 query/passage 加前缀，其余模型按配置加查询前缀。"""
        if self._family == "e5":
            prefix = "query: " if query else "passage: "
            texts = [prefix + t for t in texts]
        elif query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vectors = self._model.embed(list(texts), batch_size=self.batch_size)
        return [_l2_normalize(list(map(float, v))) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._raw_embed([text], query=True)[0]

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        return self._raw_embed(items, query=False)


class RemoteEmbeddings:
    """OpenAI 兼容 /embeddings 接口。

    典型对象：
      - 本机/内网自建：vLLM（`vllm serve Qwen/Qwen3-Embedding-8B --task embed`）、
        llama.cpp（`llama-server --embeddings`）、Xinference、TEI 等；
      - 云网关：OpenAI、硅基流动、阿里云百炼等。
    base_url 需写到 `/v1`（例如 http://127.0.0.1:8001/v1）；留空则沿用
    OPENAI_BASE_URL（比赛网关，注意它实测没有 /embeddings，会构造失败并回退）。

    query_prefix 只作用于检索侧（embed_query），文档侧不加——Qwen3-Embedding
    官方用法即为「文档裸文本 + 查询加 Instruct 前缀」。
    """

    backend = "remote"
    is_semantic = True

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        base_url: str = "",
        api_key: str = "",
        query_prefix: str = "",
        timeout: int = 60,
        dimensions: int = 0,
    ):
        try:
            from langchain_openai import OpenAIEmbeddings  # pyright: ignore[reportMissingImports]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 langchain-openai，无法使用远程 Embedding") from e

        self.model_name = model_name
        self.batch_size = max(1, int(batch_size or 16))
        # 允许在 .env 里写 \n，代码还原成真正的换行
        self.query_prefix = (query_prefix or "").replace("\\n", "\n")
        self.endpoint = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")

        kwargs: dict[str, Any] = {
            "model": model_name,
            "chunk_size": self.batch_size,
            # check_embedding_ctx_length=False：直接发原始字符串，
            # 避免网关不认 tiktoken 切分后的 token 数组（常见 400/404）
            "check_embedding_ctx_length": False,
            "timeout": max(5, int(timeout or 60)),
        }
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        if api_key:
            kwargs["api_key"] = api_key
        # 部分模型支持降维输出（如百炼 text-embedding-v4 可选 2048/1536/1024/768/...）
        if dimensions and int(dimensions) > 0:
            kwargs["dimensions"] = int(dimensions)
        self._inner = OpenAIEmbeddings(**kwargs)

        # 探测：同时验证接口可用性与维度（vLLM 首次加载模型可能较慢，超时已放宽）
        probe = self.embed_query("dimension probe")
        self.dim = len(probe)

    # ------------------------------------------------------------------
    # 限流重试：百炼等云服务对免费/低等级账号有 QPS 限制（text-embedding-v4
    # 单批 ≤10 条），全库重建 7000+ 块 = 数百次请求，429 属于常态而非异常。
    # 仅对可重试错误退避；401/404 这类配置错误立即抛出，避免白等。
    # ------------------------------------------------------------------
    _RETRY_DELAYS = (2, 5, 10, 20, 40)
    _RETRYABLE = (
        "429",
        "rate",
        "limit",
        "timeout",
        "timed out",
        "connection",
        "temporarily",
        "503",
        "502",
        " overloaded",
    )

    def _retry_call(self, fn, *args, **kwargs):
        for attempt, delay in enumerate((0,) + self._RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                retriable = any(k in msg for k in self._RETRYABLE)
                if not retriable or attempt == len(self._RETRY_DELAYS):
                    raise
                print(
                    f"[Embedding] 请求被限流/超时（{str(e)[:120]}），"
                    f"{delay}s 后重试 {attempt + 1}/{len(self._RETRY_DELAYS)}",
                    flush=True,
                )

    def embed_query(self, text: str) -> list[float]:
        payload = f"{self.query_prefix}{text}" if self.query_prefix else text
        return _l2_normalize(self._retry_call(self._inner.embed_query, payload))

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        items = [str(t) for t in texts]
        if not items:
            return []
        vectors = self._retry_call(self._inner.embed_documents, items)
        return [_l2_normalize(v) for v in vectors]


def _build_remote() -> RemoteEmbeddings:
    return RemoteEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        base_url=config.EMBEDDING_BASE_URL,
        api_key=config.EMBEDDING_API_KEY,
        query_prefix=config.EMBEDDING_QUERY_PREFIX,
        timeout=config.EMBEDDING_TIMEOUT,
        dimensions=config.EMBEDDING_DIMENSIONS,
    )


def _build_local() -> LocalOnnxEmbeddings:
    model = config.EMBEDDING_LOCAL_MODEL or DEFAULT_LOCAL_MODEL
    threads = config.EMBEDDING_THREADS or None
    return LocalOnnxEmbeddings(
        model_name=model,
        cache_dir=config.EMBEDDING_CACHE_DIR,
        threads=threads,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        allow_download=config.EMBEDDING_ALLOW_DOWNLOAD,
        query_prefix=config.EMBEDDING_LOCAL_QUERY_PREFIX,
    )


def build_embeddings(backend: str | None = None, verbose: bool = True, strict: bool = False) -> Any:
    """按配置构造 Embedding；默认逐级回退，最终一定返回可用对象。

    backend 显式指定时（auto/remote/local/hash）优先于 config.EMBEDDING_BACKEND，
    供 `reindex_embeddings.py` 的 --backend 参数复用。

    strict=True 时禁用哈希兜底（重建向量库/连通性自检必须用）：
    否则远程服务没起来会静默回退 1536 维哈希向量，把整个库写成低精度向量。
    """
    backend = (backend or config.EMBEDDING_BACKEND or "auto").strip().lower()

    if backend not in ("auto", "remote", "local", "hash"):
        if verbose:
            print(f"[Embedding] 未知后端 {backend!r}，按 auto 处理")
        backend = "auto"

    if backend == "hash":
        return LocalHashEmbeddings()

    # auto / remote / local 的候选顺序
    candidates: list[str] = []
    if backend == "remote":
        candidates = ["remote"]
    elif backend == "local":
        candidates = ["local"]
    else:
        # 配置了模型 ID 就先试远程（Key 可来自 EMBEDDING_API_KEY / config_local.py，
        # 也可能是本机无鉴权的 vLLM 服务，因此不再强制要求 OPENAI_API_KEY）
        if config.EMBEDDING_MODEL:
            candidates.append("remote")
        candidates.append("local")
    if not strict:
        candidates.append("hash")

    last_error = ""
    for name in candidates:
        try:
            if name == "remote":
                if not config.EMBEDDING_MODEL:
                    raise RuntimeError("未配置 EMBEDDING_MODEL")
                emb = _build_remote()
                if verbose:
                    print(
                        f"[Embedding] 使用远程语义模型 {emb.model_name}"
                        f"（{emb.dim} 维，{emb.endpoint or 'OPENAI_BASE_URL'}）"
                    )
            elif name == "local":
                emb = _build_local()
                if verbose:
                    print(
                        f"[Embedding] 使用本地 ONNX 语义模型 {emb.model_name}"
                        f"（{emb.dim} 维，fastembed/{_onnx_provider()}）"
                    )
            else:
                emb = LocalHashEmbeddings()
                if verbose:
                    if last_error:
                        print(
                            f"[Embedding] 语义模型不可用（{last_error}），"
                            "已回退本地哈希向量：仅支持词面匹配，检索精度较低。"
                        )
                    else:
                        print("[Embedding] 使用本地哈希向量（词面匹配，精度较低）")
            return emb
        except Exception as e:  # noqa: BLE001 - 逐个后端尝试，失败原因汇总后提示
            last_error = f"{name}: {e}"
            if verbose:
                print(f"[Embedding] {name} 后端不可用：{e}")

    if strict:
        raise RuntimeError(f"Embedding 后端不可用：{last_error}")
    return LocalHashEmbeddings()


def _onnx_provider() -> str:
    """当前 ONNX Runtime 使用的执行提供者（日志用）。"""
    try:
        import onnxruntime as ort  # pyright: ignore[reportMissingImports]

        providers = ort.get_available_providers()
        for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
            if p in providers:
                return p.replace("ExecutionProvider", "")
    except Exception:
        pass
    return "CPU"


def describe(embeddings: Any) -> dict[str, Any]:
    """把后端信息整理成字典，供 /api/health、日志与前端展示。"""
    backend = getattr(embeddings, "backend", "hash")
    semantic = bool(getattr(embeddings, "is_semantic", False))
    label = {
        "remote": "远程语义向量",
        "local": "本地 ONNX 语义向量",
        "hash": "本地哈希向量（词面匹配）",
    }.get(backend, backend)
    return {
        "backend": backend,
        "model": getattr(embeddings, "model_name", ""),
        "dim": int(getattr(embeddings, "dim", 0) or 0),
        "semantic": semantic,
        "label": label,
        "cross_lingual": semantic,
        # 远程后端暴露实际地址，方便确认连的是本机 vLLM 还是比赛网关
        "endpoint": getattr(embeddings, "endpoint", ""),
        "query_prefix": bool(getattr(embeddings, "query_prefix", "")),
    }


if __name__ == "__main__":
    # 自检：`python embeddings.py [auto|remote|local|hash]`
    # 指定后端时走严格模式（不回退哈希），用来确认自建服务是否真的连通。
    forced = sys.argv[1].strip().lower() if len(sys.argv) > 1 else None
    try:
        emb = build_embeddings(backend=forced, strict=bool(forced))
    except Exception as e:  # noqa: BLE001
        print(f"\n[失败] {e}")
        print(
            "排查顺序：\n"
            "  1) 服务是否已启动（curl http://127.0.0.1:8001/v1/models）\n"
            "  2) EMBEDDING_BASE_URL 是否写到 /v1\n"
            "  3) EMBEDDING_MODEL 是否与启动参数 --served-model-name 完全一致\n"
            "  4) 首次加载大模型较慢，调大 EMBEDDING_TIMEOUT"
        )
        sys.exit(1)
    info = describe(emb)
    print("\n[Embedding 自检]", info)
    pairs = [
        ("如何提升小样本目标检测的精度？", "How to improve few-shot object detection accuracy?"),
        ("骨架动作识别", "skeleton-based action recognition"),
        ("今天的天气", "graph convolutional network"),
    ]
    for a, b in pairs:
        va, vb = emb.embed_query(a), emb.embed_query(b)
        sim = sum(x * y for x, y in zip(va, vb))
        print(f"  cos({a!r}, {b!r}) = {sim:.3f}")
    if not info["semantic"]:
        print("\n提示：当前为哈希兜底，请检查 pip install fastembed 或模型缓存目录。")
        sys.exit(0)
