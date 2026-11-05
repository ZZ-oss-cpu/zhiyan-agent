# config.py
# ============================================================
# 项目集中配置入口：**本文件不再存放任何密钥**，可直接公开。
#
# 取值优先级（高 -> 低）：
#   1. 系统环境变量            —— 评审 / CI / Docker 环境用这个
#   2. config_local.py         —— 本地开发用，已在 .gitignore 中，不会进仓库
#   3. .env（KEY=VALUE）       —— 同上，被 .gitignore 忽略
#   4. 本文件中的默认值        —— 只放非敏感默认值（模型名、间隔、开关）
#
# 首次使用（二选一）：
#   A. 复制 .env.example 为 .env，填入 OPENAI_API_KEY / OPENAI_BASE_URL；
#   B. 复制 config_local.example.py 为 config_local.py，填入同样内容。
# 其余代码无需改动，连通性与改动前完全一致。
#
# 说明：早期版本把 Key 直接写在本文件里，现已迁出；历史 Key 应在控制台轮换。
# ============================================================
from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # 仓库根目录（src/ 的上级）


def _read_dotenv(path: str) -> dict[str, str]:
    """极简 .env 解析（KEY=VALUE），不引入 python-dotenv 依赖。"""
    data: dict[str, str] = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    data[key] = val
    except Exception:
        pass
    return data


def _read_local_module() -> dict[str, object]:
    """读取可选的 config_local.py（gitignored），只认全大写常量。"""
    data: dict[str, object] = {}
    try:
        import config_local as _cl  # noqa: F401  本地私密配置，缺失时静默跳过
    except Exception:
        return data
    for key in dir(_cl):
        if not key.isupper():
            continue
        val = getattr(_cl, key)
        if isinstance(val, (str, int, float, bool)):
            data[key] = val
    return data


_LOCAL: dict[str, object] = {
    **_read_dotenv(os.path.join(ROOT_DIR, ".env")),  # .env 固定在仓库根目录
    **_read_local_module(),  # config_local.py 优先级高于 .env
}


def _get(name: str, default: str = "") -> str:
    """环境变量 > 本地私密配置 > 默认值；空字符串视为未设置。"""
    env_val = os.environ.get(name)
    if env_val is not None and str(env_val).strip() != "":
        return str(env_val)
    if name in _LOCAL and str(_LOCAL[name]).strip() != "":
        return str(_LOCAL[name])
    return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(str(_get(name, str(default))).strip())
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = str(_get(name, "")).strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on", "y")


# ---- 必填：API Key（留空则自动使用离线 Mock 模式）----
# 不要再把真实 Key 写在这里；填到 .env 或 config_local.py。
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")

# ---- 选填：国产 OpenAI 兼容接口地址（用 OpenAI 官方则留空）----
# DeepSeek 示例: "https://api.deepseek.com/v1"
# 注意：只填到 /v1 为止，客户端会自动拼接 /chat/completions 等路径
OPENAI_BASE_URL = _get("OPENAI_BASE_URL", "")

# ---- 对话模型名称（按平台实际提供的模型 ID 填写，大小写敏感）----
LLM_MODEL = _get("LLM_MODEL", "DeepSeek-V4-Flash")

# ---- Embedding 模型名称（远程 OpenAI 兼容 /embeddings 接口的模型 ID）----
# 留空 = 不尝试远程，直接使用本地 ONNX 语义模型（推荐，无需额外 Key）。
# 本地自建服务（vLLM / llama.cpp / Xinference / TEI）填其 served-model-name 即可。
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "")

# ---- 远程 Embedding 服务地址与密钥（与聊天模型的网关解耦，可指向本机 8001 端口）----
# 留空则回退到 OPENAI_BASE_URL / OPENAI_API_KEY（沿用比赛网关）。
# 例：EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
EMBEDDING_BASE_URL = _get("EMBEDDING_BASE_URL", "")
EMBEDDING_API_KEY = _get("EMBEDDING_API_KEY", "")
# 查询侧指令前缀（仅 embed_query 加，文档侧不加）。
# Qwen3-Embedding 官方建议：Instruct: {任务描述}\nQuery:
# .env 中可写 \n 转义，代码会还原为换行。
EMBEDDING_QUERY_PREFIX = _get("EMBEDDING_QUERY_PREFIX", "")
# 单次请求超时（秒）：8B 模型 + CPU 推理可能很慢，按需放大
EMBEDDING_TIMEOUT = _get_int("EMBEDDING_TIMEOUT", 60)
# 输出向量维度（0 = 用服务端默认）。百炼 text-embedding-v4 可选 2048/1536/1024/768/512/256/128/64
EMBEDDING_DIMENSIONS = _get_int("EMBEDDING_DIMENSIONS", 0)

# 本地 fastembed 模型的查询前缀（切到 Qwen3-Embedding 时按官方格式填，MiniLM 留空）
EMBEDDING_LOCAL_QUERY_PREFIX = _get("EMBEDDING_LOCAL_QUERY_PREFIX", "")

# ---- 公式视觉 LaTeX 还原（pix2tex / LaTeX-OCR，可选）----
# auto = 已安装就用（首次调用会下载约 100MB 权重）；off = 只用规则转换（零依赖）
ENABLE_LATEX_OCR = _get("ENABLE_LATEX_OCR", "auto").strip().lower()

# ---- Embedding 后端：auto（默认）/ remote / local / hash ----
#   auto  ：配置了 EMBEDDING_MODEL 且有 Key 时先试远程，失败自动回退本地；
#   local ：fastembed + ONNX Runtime，CPU 可跑、可完全离线；
#   hash  ：词元哈希兜底（无网络无模型时仍能跑通流程，但只有词面匹配能力）。
EMBEDDING_BACKEND = _get("EMBEDDING_BACKEND", "auto")

# ---- 本地语义模型（多语言，中英跨语言检索；首次运行自动下载到缓存目录）----
# 备选（改这里即可，维度不同需重跑 reindex_embeddings.py）：
#   BAAI/bge-small-zh-v1.5                 512 维，纯中文更快
#   intfloat/multilingual-e5-large         1024 维，效果更好但更慢更大
#   sentence-transformers/all-MiniLM-L6-v2 384 维，纯英文
EMBEDDING_LOCAL_MODEL = _get(
    "EMBEDDING_LOCAL_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# 模型缓存目录（默认仓库根目录 models/，已在 .gitignore 中；填相对路径也按仓库根解析）
_EMB_CACHE = _get("EMBEDDING_CACHE_DIR", "models")
EMBEDDING_CACHE_DIR = (
    _EMB_CACHE if os.path.isabs(_EMB_CACHE) else os.path.join(ROOT_DIR, _EMB_CACHE)
)
# 是否允许自动下载模型：答辩/断网演示设为 false，只用本地缓存
EMBEDDING_ALLOW_DOWNLOAD = _get_bool("EMBEDDING_ALLOW_DOWNLOAD", True)
# ONNX 推理线程数（0 = 交给 onnxruntime 自动决定）
EMBEDDING_THREADS = _get_int("EMBEDDING_THREADS", 0)
# 批量嵌入大小（入库时一次送多少段文本）
EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE", 16)

# ---- Qdrant 向量库（留空则沿用 source_tracing.py 的默认 ./qdrant_db）----
QDRANT_PATH = _get("QDRANT_PATH", "")
QDRANT_COLLECTION = _get("QDRANT_COLLECTION", "academic_papers")

# ---- 语义判重阈值（0 = 按向量类型自动：语义 0.88 / 哈希 0.65）----
# 换用不同 Embedding 模型后如误报偏多，可上调到 0.90；漏报偏多则下调到 0.84
DEDUP_SIM_THRESHOLD = _get("DEDUP_SIM_THRESHOLD", "0")

# ---- TOOL 分支：arXiv 最新论文定期抓取 ----
# 定期抓取总开关（断网环境 / 答辩演示时可设为 False，避免后台反复抓取失败刷日志）
FETCH_ENABLED = _get_bool("FETCH_ENABLED", True)
# 检索关键词（留空则自动使用知识库内论文标题提炼的偏好关键词）
ARXIV_KEYWORDS = _get("ARXIV_KEYWORDS", "skeleton-based action recognition")
# 抓取间隔（小时）
FETCH_INTERVAL_HOURS = _get_int("FETCH_INTERVAL_HOURS", 6)

# ---- 网络代理（访问 arXiv / GitHub 需要时填写，不需要则留空）----
# 仅 arxiv_tool 的学术 API 请求走此代理，不会注入全局环境变量，
# 因此不影响 LLM 内网接口的直连。
# 注意：代理软件没启动时填这里的端口会报 [WinError 10061] 目标计算机积极拒绝，
#       此时请留空（直连）或先把代理软件打开并核对端口。
HTTPS_PROXY = _get("HTTPS_PROXY", "")  # 例：开了 Clash 再填 "http://127.0.0.1:7897"

# ---- 学术 API 的 TLS 证书校验降级 ----
# 部分校园网 / 安全软件会用自己的证书解密 HTTPS，导致
# "certificate verify failed" 而抓不到 arXiv / PubMed。
# True = 仅在证书校验失败时自动降级为不校验并重试一次（只影响公开学术元数据请求）；
# False = 严格校验，遇到上述网络会直接失败。
ALLOW_INSECURE_TLS = _get_bool("ALLOW_INSECURE_TLS", True)


def apply():
    """把配置注入环境变量（不覆盖已存在的系统环境变量）。

    下游（langchain_openai / LangGraph）统一从环境变量读 Key，所以这里保持
    「有值才写、已有不覆盖」的行为，改动配置来源后连通性不变。
    """
    if OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if OPENAI_BASE_URL and not os.environ.get("OPENAI_BASE_URL"):
        # 自动纠偏：误填完整端点路径时截断到 /v1，避免路径重复拼接导致 404
        base = OPENAI_BASE_URL.rstrip("/")
        for suffix in ("/chat/completions", "/embeddings", "/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        os.environ["OPENAI_BASE_URL"] = base
    if LLM_MODEL and not os.environ.get("LLM_MODEL"):
        os.environ["LLM_MODEL"] = LLM_MODEL
