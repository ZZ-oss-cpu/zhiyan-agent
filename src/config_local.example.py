# config_local.example.py —— 本地私密配置模板
#
# 用法：复制本文件为 config_local.py，填入自己的 Key。
#       config_local.py 已在 .gitignore 中，不会被提交。
#
# 也可以不用这个文件，改用 .env（见 .env.example），二者等效；
# 或直接设置系统环境变量（优先级最高，适合评审 / Docker 环境）。

OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_BASE_URL = ""  # 用 OpenAI 官方留空；DeepSeek 填 "https://api.deepseek.com/v1"
LLM_MODEL = "gpt-4o-mini"  # 按平台实际模型 ID 填写

# 可选（不填则用 config.py 里的默认值）
# Embedding：默认走本地 ONNX 多语言语义模型（无需 Key，首次自动下载约 240MB）
# EMBEDDING_BACKEND = "auto"          # auto / remote / local / hash
# EMBEDDING_MODEL = "text-embedding-3-small"   # 仅当平台实现了 /embeddings 时填
# ---- 自建 Qwen3-Embedding（vLLM / llama.cpp）示例：本机调 GPU 机器 ----
# EMBEDDING_BACKEND = "remote"
# EMBEDDING_MODEL = "Qwen3-Embedding-8B"
# EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"
# EMBEDDING_API_KEY = "EMPTY"
# EMBEDDING_QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant academic passages\nQuery:"
# EMBEDDING_TIMEOUT = 120
# ---- 阿里云百炼（Qwen3-Embedding 系列，API 模型 ID 为 text-embedding-v4）----
# EMBEDDING_BACKEND = "remote"
# EMBEDDING_MODEL = "text-embedding-v4"
# EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# EMBEDDING_API_KEY = "sk-xxxxxxxx"        # bailian.console.aliyun.com 创建
# EMBEDDING_QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant academic passages\nQuery:"
# EMBEDDING_BATCH_SIZE = 10                # 百炼单批最多 10 条，勿调大
# EMBEDDING_DIMENSIONS = 0                 # 0 = 默认 1024；可选 2048/1536/768/512...
# EMBEDDING_TIMEOUT = 60
# ---- 本机 CPU 跑 Qwen3-Embedding-0.6B（1024 维，fastembed，约 1.1GB）----
# EMBEDDING_BACKEND = "local"
# EMBEDDING_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B-Q"
# EMBEDDING_LOCAL_QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant academic passages\nQuery:"
# EMBEDDING_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_CACHE_DIR = "models"      # 相对仓库根目录
# EMBEDDING_ALLOW_DOWNLOAD = False    # 断网演示设为 False（只用已有缓存）
# DEDUP_SIM_THRESHOLD = 0             # 0 = 按向量类型自动（换模型后建议 0.90）
# HTTPS_PROXY = "http://127.0.0.1:7897"
# FETCH_ENABLED = True
# FETCH_INTERVAL_HOURS = 6
# ALLOW_INSECURE_TLS = True
