# 智研 Agent —— 项目运行指南

基于 LangGraph 的全流程科研智能助手（方案一：Multi-Agent 分级工作流）。

**能力总览**：论文 RAG 溯源问答（行级精准高亮）、引用图谱基石挖掘（真实引用网络）、多论文遍历对比、Future Work / 研究空白分析、语义去重与版本演进、学术写作辅助（Idea → 论文框架 + 无幻觉 References + TikZ/Graphviz/Mermaid/Matplotlib 拓扑图脚本 + CSV 表格/折线/柱状/三维雷达图 + 自适应图注 + 中译英翻译）、整篇翻译（原位排版 + Markdown 导出）、中英对照翻译与联动滚动、历史对话存档与答案回补、arXiv/PubMed 文献收集闭环、复合任务自动编排。

**Web 前端地址**：`http://localhost:8000`

---

## 一、项目结构

```
agent/
├── config.py                 # 【集中配置】API Key / 模型 / 网络代理 / 抓取关键词与间隔
├── Agent.py                  # LangGraph 工作流：RAG / GRAPH / VISUAL / WRITE / DIAGRAM / TOOL / PLAN 七分支
├── Source Tracing.py         # RAG：PyMuPDF 提取带页码坐标文本块 + Qdrant 向量库 + 检索重排
├── embeddings.py             # Embedding 工厂：远程 / 本地 ONNX 语义模型 / 哈希兜底（含维度探测）
├── reindex_embeddings.py     # 换模型后的向量库重建（备份 → 临时集合核对 → 原子替换）
├── graph and generation.py   # 图谱：PageRank + 介数中心性 + 真实引用边构建 + Research Gap
├── writing.py                # 学术写作辅助：论文框架 / 拓扑图脚本 / 中译英 / 图注 / DOI 解析
├── visualization.py          # 可视化沙箱：CSV -> 表格/折线/柱状/三维雷达图 + 自适应图注（Matplotlib）
├── version_dedup.py          # 语义版本去重与 v1->v7 Diff 对比
├── arxiv_tool.py             # arXiv 检索 / PDF 下载 + GitHub 复现性查询
├── scholar_sources.py        # 多来源统一检索（arXiv/PubMed/自定义API）+ PDF 直链提取 + 标识符解析
├── bibparse.py               # RIS / BibTeX 批量解析
├── citation_quality.py       # 引用含金量：五维确定性打分 + 应付式引用识别（模块四）
├── dedup_service.py          # 语义查重与 Human-in-the-Loop 版本检查
├── requirements.txt          # 依赖清单
├── README.md                 # 本指南
└── front_window/             # 【Web 前端】
    ├── app.py                # FastAPI 服务：对话 / 知识库 / 工具接口 / 定时抓取 / 预留接口
    ├── index.html            # 单页前端：左侧导航 + Agent 首页 / 知识库 / 科研问答 / 项目 / 工具
    ├── pdf.min.js            # pdf.js 本地文件（离线可用）
    ├── pdf.worker.min.js     # pdf.js Worker
    ├── uploads/              # 导入的 PDF 原文件
    ├── qdrant_db/            # 本地向量库（含 page_num / bbox 锚点）
    ├── papers.json           # 论文登记簿（含版本、演进 Diff、元数据）
    ├── fetch_settings.json   # 抓取设置（关键词 / 来源 / 自定义数据源列表 + 多选启用的 id 列表）
    └── latest_papers.json    # 最近一次定期抓取结果
```

---

## 二、环境准备

### 1. Python 版本

要求 **Python >= 3.10**（开发环境为 Python 3.14）。

### 2. 安装依赖

```powershell
cd e:\比赛-综合\agent
pip install -r requirements.txt
```

> - 最小（离线演示）：`langgraph langchain-core networkx pandas matplotlib`
> - 真实 RAG：`qdrant-client pymupdf langchain-openai`
> - Web 前端：`fastapi uvicorn python-multipart`

### 3. 激活虚拟环境（必须，否则 `uvicorn` 会报「No module named uvicorn」）

本项目的虚拟环境是 **venv**（`e:\比赛\agent\.venv`，靠 `pyvenv.cfg` 识别），**不是 conda 环境**。

PowerShell：

```powershell
cd e:\比赛\agent
.\.venv\Scripts\Activate.ps1
```

cmd：

```bat
cd /d e:\比赛\agent
.venv\Scripts\activate.bat
```

Git Bash / MSYS2：

```bash
cd /e/比赛/agent
source .venv/Scripts/activate
```

激活成功的标志：提示符前出现 `(.venv)`，且 `python -m uvicorn --version` 能打印版本号。

不想激活也可以，直接指定 venv 的解释器（效果等价，最不容易出错）：

```powershell
E:\比赛\agent\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

**激活报错对照**：

| 现象 | 原因 | 处理 |
|---|---|---|
| `conda activate .env` 报 `Could not find conda environment` / `EnvironmentNameNotFound` | `conda activate` 只认 **conda 环境**（需存在 `conda-meta\` 目录）；`.venv` 是 `python -m venv` 建的，conda 不识别。另外项目里**没有 `.env` 目录**（`.env` 通常只是存环境变量的 dotenv 文件，不是虚拟环境） | 改用 `.\.venv\Scripts\Activate.ps1` |
| `conda : 无法将"conda"项识别为 cmdlet…` | conda 未加入当前 PowerShell 的 PATH | 用「Anaconda Prompt」，或先执行 `conda init powershell` 再重开终端 |
| `C:\Python314\python.exe: No module named uvicorn` | 终端里的 `python` 指向**系统解释器**，依赖都装在 `.venv` 里 | 先激活 venv，或用 `E:\比赛\agent\.venv\Scripts\python.exe` |
| `Activate.ps1 无法加载…因为在此系统上禁止运行脚本` | PowerShell 执行策略限制 | 先 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`（只影响当前窗口） |

> 若确实想用 conda：`conda create -n agent python=3.12` 后重新 `pip install -r requirements.txt`（torch 等体积大，一般没必要）。日常开发建议继续用 `.venv`。

---

## 三、配置（密钥已与代码分离）

**`config.py` 里不再存放任何密钥**，可以直接公开。配置按以下优先级读取（高 → 低）：

```
系统环境变量  >  config_local.py  >  .env 文件  >  config.py 默认值
```

**首次使用（二选一）**：

```powershell
# 方式 A：复制环境变量模板（推荐，评审/部署环境也用它）
copy .env.example .env        # 然后编辑 .env 填入 OPENAI_API_KEY 等

# 方式 B：复制本地私密配置模板
copy config_local.example.py config_local.py
```

`.env`、`config_local.py` 均已写入 `.gitignore`，**不会**被提交到仓库；系统环境变量优先级最高，方便 CI / Docker / 评审环境直接注入。

| 配置项 | 说明 |
|---|---|
| `OPENAI_API_KEY` | 对话模型 Key（留空 → 离线 Mock） |
| `OPENAI_BASE_URL` | 国产兼容接口地址（如 DeepSeek），官方 API 留空 |
| `LLM_MODEL` | 对话模型 ID |
| `EMBEDDING_BACKEND` | `auto`（默认）/ `remote` / `local` / `hash`。auto 会先试远程、失败自动回退本地 ONNX |
| `EMBEDDING_MODEL` | 远程向量模型 ID（**只有平台实现了 `/v1/embeddings` 才填**；留空直接用本地模型；自建服务填 `--served-model-name`） |
| `EMBEDDING_BASE_URL` | 远程 Embedding 服务地址，**必须写到 `/v1`**（如 `http://127.0.0.1:8001/v1`）；留空沿用 `OPENAI_BASE_URL` |
| `EMBEDDING_API_KEY` | 远程服务密钥（本地无鉴权的 vLLM 填 `EMPTY` 即可） |
| `EMBEDDING_QUERY_PREFIX` | 查询侧指令前缀（Qwen3 官方 `Instruct: ...\nQuery:`，只加在检索侧） |
| `EMBEDDING_TIMEOUT` | 单次请求超时秒数（8B 首次加载建议 180） |
| `EMBEDDING_DIMENSIONS` | 输出向量维度（0 = 服务端默认；百炼 text-embedding-v4 可选 2048~64） |
| `EMBEDDING_LOCAL_MODEL` | 本地语义模型（默认 `paraphrase-multilingual-MiniLM-L12-v2`，384 维；CPU 可换 `Qwen/Qwen3-Embedding-0.6B-Q`，1024 维） |
| `EMBEDDING_CACHE_DIR` | 本地模型缓存目录（默认仓库根 `models/`，已 gitignore） |
| `ENABLE_LATEX_OCR` | 公式视觉 LaTeX 还原：`auto`（默认，装了 pix2tex 就用）/ `off`（只用规则转换）。pix2tex 为可选依赖（`pip install pix2tex`，含 torch CPU 约 1GB），未安装时自动回退 Unicode 规则还原 |
| `EMBEDDING_ALLOW_DOWNLOAD` | 断网演示设 `false`：只读缓存，缺模型时回退哈希向量 |
| `HTTPS_PROXY` | 网络代理，**仅 arXiv / GitHub 学术请求走此代理**，不影响 LLM 内网接口。**留空 = 直连**；开了代理软件再填实际端口（端口没开时会报 `WinError 10061`） |
| `ALLOW_INSECURE_TLS` | 默认 `True`：证书校验失败时（校园网 / 安全软件替换证书）自动降级为不校验并重试一次，仅影响公开学术元数据请求 |
| `ARXIV_KEYWORDS` | 定期抓取关键词（留空 → 用知识库标题提炼的偏好词） |
| `FETCH_INTERVAL_HOURS` | 定期抓取间隔（默认 6 小时） |

> RAG 默认使用**本地 ONNX 语义模型**（fastembed + onnxruntime，无需 API Key、无需 torch、可离线），
> 384 维、支持中英跨语言检索；平台若实现了 `/v1/embeddings`，把模型 ID 填进 `EMBEDDING_MODEL` 即优先走远程。
> 两者都不可用时才回退哈希向量（词面匹配 + 词面重排 + LLM 中文改写英文检索词）。
>
> 安全说明：历史版本曾把真实 Key 直接写在 `config.py` 中，现已全部迁出；如果你在旧版本里写过 Key，请在服务平台**轮换（作废重发）**该 Key，仅删除文件不能使其失效。

---

## 四、运行方式

### 1. Web 平台（主入口，推荐）

```powershell
cd e:\比赛\agent\front_window
..\.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

> 若已按第二节第 3 条激活了虚拟环境，直接 `uvicorn app:app --port 8000` 即可；未激活时用上面这种写法，
> 否则会命中系统 Python 报 `No module named uvicorn`。首次启动需 30 秒左右加载依赖，看到
> `Uvicorn running on http://127.0.0.1:8000` 才算就绪。

浏览器打开 **http://localhost:8000**。前端为左侧导航 + 五个视图的单页应用（浅色学术风）：

| 视图 | 说明 |
|---|---|
| **Agent（首页）** | 渐变标题 + 任务输入框 + 任务模式胶囊（灵感发现 / 文献综述 / 论文复现 / 论文评审 / Idea → 论文框架 / 架构拓扑图 / CSV 数据可视化）+ 「案例回放」论文卡片墙（取知识库真实论文，可一键套用提问） |
| **知识库** | 表格化论文列表（标题 / 期刊年份 / 创建者 / 状态 / 操作），搜索框实时过滤，多选限定检索范围，版本徽章 + 演进 Diff 弹窗；上传下拉菜单（文件 / 文件夹 / 粘贴 URL / 标识符 / RIS·BibTeX / CSV），支持拖拽导入 |
| **科研问答** | 居中提问框（思考强度、已选 N 项）+ 示例问题，输入内容后发送箭头由灰转蓝（模板/案例/划词提问等程序化填充同样生效）；发送后切换为对话流，右侧「上下文面板」展示思考日志、引用图谱、溯源锚点与生成图表（可收起） |
| **项目** | 科研项目卡片（本地保存，后端 `/api/projects` 接入后自动上云），排序 / 归档 / 重命名；「进入」即把项目论文设为检索范围；**「管理论文」弹窗可随时勾选知识库论文收入 / 移出项目**（支持标题过滤、全选 / 清空，保存后若为当前工作空间则问答范围同步更新）；顶部知识库存储用量条。**新建入口为样式化下拉菜单（两个选项）**：侧边栏「选择工作空间 +」或项目页「新建项目」→ 弹出「**＋ 新建项目**」（输入名称创建 → 自动弹「管理论文」勾选知识库论文，知识库已勾选的论文自动预选 → 保存后自动进入问答页可直接提问）与「**📁 打开本地文件夹**」（**直接弹系统文件夹选择框**，选中电脑磁盘里的文件夹后批量上传入库，新入库论文**自动收入当前工作空间**，即可提问）→ 应用内输入弹窗（回车 / 确定提交，替代浏览器原生 prompt）；重命名同样走该弹窗。侧边栏工作空间条目 **hover 显示 × 可删除**（确认后删除，不删除知识库论文），项目卡片上同样有「删除」 |
| **工具** | TrueCite **引用含金量筛选**（粘贴编号参考文献 / 上传 Bib·RIS → 五维打分 + 应付式引用自动剔除 + 保留条目一键复制/导入）、论文插图转 PPT、论文图表转 Excel、数据转科研图、中译英学术翻译（粘贴中文摘要/全文 → 学术英文，公式/变量/引用编号保留） |

各视图保留的原有能力：

| 功能 | 说明 |
|---|---|
| 对话路由 | 自动路由 RAG / GRAPH / VISUAL / WRITE / DIAGRAM / TOOL / PLAN，分支徽章显示在回答上 |
| 在线阅读 | 点击论文打开 pdf.js 阅读器；点击锚点**跳页并逐行精准高亮**原文（高亮块用 `Range.getClientRects()` 按文字真实渲染矩形绘制，与正文等长、不越出行宽，行内跨文本块自动合并） |
| 划词提问 | PDF 内选中文字 → 浮动按钮 → 自动切到科研问答并填入聊天框 |
| 划词翻译 | PDF 内选中文字 → 「🌐 翻译」浮动按钮 → 原地弹出中文译文（公式/代码/引用编号保留原文） |
| 中英对照翻译 | 阅读器顶栏「中英对照」开关：左侧原版 PDF、右侧中文译文面板；**按页懒加载**（滚到哪页翻哪页，避免整本调用 LLM），公式/代码/引用编号不译；**双向联动滚动**——左栏滚动右栏自动跟随到对应页，右栏滚动左栏同步翻页。扫描件同样可用（走同一套 OCR chunk）。窗口默认**适应宽度**：无论目录/翻译面板开多少个，PDF 整页自动缩放、两侧不会被裁掉；顶栏 −/＋ 可手动缩放，点「适应」恢复自动；面板间的**分隔条可拖拽**调整宽度。译文**含标题 / 摘要 / 章节标题等短块**（阅读器取数用 `min_len=4`，入库检索仍用 30，互不影响） |
| 整篇翻译 | 阅读器顶栏「⇄ 整篇翻译」按钮：抽取全文结构（标题层级/正文/公式/表格/图注/参考文献）→ 分块翻译 → 组装**原位排版 Markdown**（正文/标题/图注译为中文，公式/代码/参考文献保留原文，按页码锚点保留阅读顺序），一键预览 + 复制 + 下载 `.md`。**表格分级导出**：结构可信 → Markdown 表格（单元格译文）；结构存疑（列切分错位会产生碎片乱码）→ 自动嵌入该表区域截图（data URI，渲染永远正确）；三线表启发式 → 注释提示。支持英→中 / 中→英 |
| 论文框架生成 | 首页「Idea → 论文框架」胶囊 / 对话路由 WRITE 分支：由 idea 生成 Title / Abstract / Introduction / Related Work；**References 只引用知识库内真实文献**（DOI 经 Crossref 标题检索补全并缓存，无幻觉），右侧面板结构化卡片渲染 |
| 架构拓扑图脚本 | 首页「架构拓扑图」胶囊 / 对话路由 DIAGRAM 分支：一次性产出 **TikZ / Graphviz / Mermaid / Matplotlib** 四种格式脚本，代码块一键复制 |
| 历史对话 | 问答页顶栏「🕘 历史对话」：会话完整存档（回答 + 图表/框架/脚本/锚点入档，localStorage 配额不足自动逐级降级）；旧记录（只存了提问）恢复时自动从后端 LangGraph checkpoint 回补缺失回答；带结果的回答气泡可点击在右侧面板重新查看 |
| 标题树导航 | 顶栏「目录」开关：优先从 PDF 书签（`get_toc`）生成标题树；**无书签的文档（多数 arXiv/CVPR 论文）自动构建「书籍式」标题树**——以结构化信号为主：章节编号（1 / 2.1 / 3.1.1）定层级、常见无编号节名（Abstract / References 等）归一级、大号粗体行作论文大标题，配合**断行标题合并**（仅编号行 "2.2." 吸收续行、同 y 拆分行合并、居中大标题跨行合并）；用「整行粗体占比 / 行内字号混合（上标检测）/ 斜体 / 参考文献区截断 / arXiv 水印过滤」排除图注、机构名、表格数字等噪声，按层级缩进、点击跳页 |
| 引用跳转 | 顶栏「引用跳转」开关：从 PDF 内部链接（`get_links`，LaTeX hyperref 生成）提取正文引用标记位置，叠加半透明热区（虚线框），点击**无损跳转**到参考文献条目所在页；依赖文档自带内部链接，扫描件无链接时不显示 |
| 图谱 | ECharts 力导向图，库内蓝 / 外部灰 / 基石红，悬停显示 PageRank / 介数中心性 |
| 最新论文 | 顶栏「领域最新论文」，可自定义关键词与来源（arXiv/PubMed/自定义API），后台定时抓取；自定义API 支持多数据源管理、站点预设与「测试检索」，填法见**第六节** |
| 文献收集 | 导入菜单：标识符导入（DOI/arXiv/PubMed）、RIS/BibTeX 批量导入、URL 直接导入、CSV 上传 |
| 公式抽取 | 阅读器顶栏「∑ 公式」开关：**启发式识别文本层中的数学公式块**（文本短、数学符号密度高、非完整句子），侧栏列出公式与所在页，点击跳页并高亮。每条公式附 **LaTeX 还原**（默认 Unicode 规则层即时转换；点「AI 精修」走 LaTeX-OCR 视觉识别）+ **版面语义标注**（公式编号、独立/含文字、所属章节、上下文片段），LaTeX 可一键复制；翻译面板中公式块**保留原文不译**（等宽字体底纹展示）。扫描件无文本层时不支持 |
| 表格图表抽取 | 阅读器顶栏「表格图表」开关：**结构化抽取表格与图表**——表格用 `page.find_tables()`（有框线）+ `_detect_ruleless_tables`（**三线表**：`get_drawings` 水平线聚合 + 区域文本按行/列还原）还原为二维行；图表按图片块（位图）+ `_detect_vector_figures`（**矢量图**：`get_drawings` 图形聚合）识别，就近匹配图注 caption。侧栏点击展开 HTML 表格、跳页高亮，并**渲染区域缩略图预览**（公式侧栏同样有预览图）。局限：完全无线的纯文本对齐表格仍是启发式边界 |

### 2. 命令行演示

```powershell
python Agent.py                     # 七分支工作流演示（含 HITL 打断恢复）
python "graph and generation.py"    # 图谱挖掘 + Research Gap
python version_dedup.py             # 语义版本去重与 Diff
python arxiv_tool.py                # arXiv 连通性自检
python bibparse.py                  # RIS/BibTeX 解析自检
```

---

## 五、七分支工作流说明

| 分支 | 触发示例 | 行为 |
|---|---|---|
| **RAG** | 「T-MOR 这篇论文的核心方法是什么？」 | 单论文过滤检索，回答带可读溯源列表（论文/页码/相关度） |
| **GRAPH** | 「分析引用图谱，挖掘基石论文」 | 从库内参考文献区解析真实引用边 → PageRank + 介数中心性；**带 HITL 人工确认** |
| **VISUAL** | 「根据 CSV 画出 Accuracy 曲线」 | 表格 + 折线图 / 柱状图 / 三维雷达图 + 自适应学术图注 |
| **WRITE** | 「把 idea 写成论文框架 + References」 | 生成 Abstract/Intro/Related Work；References 只引知识库真实文献（DOI 经 Crossref 校验） |
| **DIAGRAM** | 「生成架构拓扑图脚本」 | 产出 TikZ / Graphviz / Mermaid / Matplotlib 四格式脚本 |
| **TOOL** | 「在 arXiv 上搜索 / 下载 / 推荐最新论文」 | 调用 arXiv + GitHub，下载模式形成「搜索→下载→入库」闭环 |
| **PLAN** | 「分析 5 篇论文方法与异同 + 构建引用关系 + 综述 + Future Work」 | 复合任务拆解为 RAG→GRAPH→SURVEY→FUTURE 依次执行后合并 |

其他自动化能力：
- **跨语言检索**：中文提问由 LLM 改写为英文检索词后检索英文论文
- **参考文献排除**：入库时标记 References 区，检索时剔除（避免引用区噪声）
- **多论文遍历**：点名多篇或「知识库/N篇/比较」等全库意图 → 逐篇检索后综合对比
- **全文级提问**：核心方法/总结等问题自动扩大 top_k 并注入摘要块
- **语义去重**：上传时查重，疑似旧版本返回，前端确认后可作为新版本替换（v1→v2）并保留演进 Diff
- **PDF 元数据识别**：上传时提取 DOI → Crossref 补全作者/期刊/年份
- **扫描件 OCR**：无文本层的 PDF 自动降级为 [RapidOCR](https://github.com/RapidAI/RapidOCR)（PaddleOCR PP-OCR 模型的 ONNX 轻量版，模型随包分发、CPU 离线可跑，`pip install rapidocr`）：整页渲染（200dpi）→ 检测+识别（置信度 < 0.5 丢弃）→ 双栏中缝排序 → 段落块聚合，产出与原生文本层**完全同构**的 chunk（含 `page_num`/`bbox` 锚点），查重、References 排除、行级高亮链路零改动复用；OCR 出的文本同样参与语义查重（实测：扫描版 Attention 与库内原生版相似度 0.91，被 409 拦截）。未安装 rapidocr 时原生 PDF 不受影响，仅扫描件返回 422 并在报错中给出安装命令。CPU 上每页约 2–6 秒，纯图扫描件首次入库会明显慢于原生 PDF

---

## 六、自定义数据源（「添加数据源」）填写指南

### 1. 这个功能在做什么

入口：**科研问答**页右上角 `🕐 领域最新论文`。面板顶部的检索栏里有 `arXiv / PubMed / 自定义API` 三个来源开关。
勾选「自定义API」后，系统会用你添加的**接口地址**去请求论文列表。

**可以同时添加多个自定义 API，并让它们一起抓取**：来源开关右侧的下拉按钮就是「数据源管理」入口 ——
点开能看到全部已添加的数据源，每行一个勾选框（勾上＝参与本次抓取，可多选，也可以全选），
行内还有「文档」链接和「删除」按钮。勾选变化会立刻保存，下一次抓取即生效。
按钮上的文字会显示当前状态：`自定义API（2 个）`（全部启用）、`1/3 个源`（部分启用）、`未选择数据源`。

抓回来的结果：

- 多个源的结果会**合并去重**后按发布时间**倒序**展示（最新在前），卡片前缀显示来源名，如 `[OpenAlex]`、`[DOAJ]`；
- 某个源失败（如限流 429、接口超时）不影响其它源，失败原因显示在列表顶部，形如 `custom[DOAJ]: HTTP Error 429`；
- 自动从返回 JSON 里找 **PDF 直链**；
- 有 PDF 的条目出现「导入知识库」按钮：下载 → 语义查重 → 解析入库 → 左侧知识库刷新；
- 没 PDF 的条目显示「无开放 PDF（可手动下载后上传）」。

所以「添加数据源」要填的不是某个网站的网址，而是**该网站返回 JSON 的 API 接口地址**。

### 2. 图文对应：每个输入框怎么填

弹窗路径：「领域最新论文」→ 检索栏的 `＋ 添加` → 弹出「添加数据源」。

| 输入框 | 必填 | 填写规则 | 例子 |
|---|---|---|---|
| **站点预设** | 否 | 下拉选常见学术站点，选中后自动填好「名称 / API 地址 / 文档地址」，一般只需再按需补 Key。选「自定义（手动填写 API 地址）」则全部手填。**预设只是模板，填进去之后仍可修改** | 选 `OpenAlex` → 地址自动变成 `https://api.openalex.org/works?search={query}&filter=from_publication_date:{from}&per-page={limit}` |
| **名称** | 否 | 自己起的别名，仅用于展示（下拉框、抓取结果的 `[名称]` 前缀）。留空自动取 URL 前 24 个字符 | `OpenAlex`、`课题组文献库` |
| **API 地址** | **是** | 必须是**返回 JSON 的接口地址**，并且带 `{query}` 占位符；系统会把关键词填进去。三条自检：① 域名是 `api.xxx` 或路径含 `/api/`，不是 `docs.`/产品介绍页；② 浏览器直接打开（把 `{query}` 换成 `python`）能看到**一堆 JSON**，不是网页；③ 不需要登录 Cookie | `https://api.openalex.org/works?search={query}&per-page={limit}` |
| **API Key** | 否 | 免 Key 的站点留空即可。填了之后系统会自动以 `Authorization: Bearer` 和 `x-api-key` 两个请求头发送（兼容 OpenAlex / Semantic Scholar / CORE 等不同约定）；如果 URL 里写了 `{api_key}` 占位符，则改为作为 URL 查询参数传递 | Semantic Scholar 的 Key：`a1b2c3...` |
| **时间范围** | 否 | 只影响 URL 里写了 `{from}` 的数据源：近 7 / 30 / 90 天或近 1 年，写入 `{from}` 的具体日期（今天 − N 天）。选「不限」时 `{from}` = `1900-01-01`（等于不限制）。**URL 里没有 `{from}` 时，这个选项不改变请求，只由站点自身默认返回** | 选「近 30 天」→ `{from}` = `2026-10-02` |
| **文档地址** | 否 | 纯粹给「查看文档」链接用，不参与抓取 | `https://docs.openalex.org/api-entities/works` |

> 「保存」和「测试检索」的区别：**测试检索不保存**，它拿当前表单内容实跑一次请求，返回「命中 N 条 / M 条有 PDF」和前 3 条标题，用来验证地址填得对不对；确认没问题再点「保存」写入配置。
>
> 注意区分两个保存：「添加数据源」弹窗里的**保存**（把这条源写进配置，并自动勾选「自定义API」、设为当前源）；
> 面板检索栏的**保存设置**（保存关键词 + 三个来源开关）。改完关键词要点后者才生效。

### 3. 占位符（URL 里可以写哪些变量）

| 占位符 | 会被替换成 | 展开示例 |
|---|---|---|
| `{query}` | 当前关键词，已做 URL 编码 | `skeleton%20action%20recognition` |
| `{limit}` / `{max_results}` / `{per_page}` | 期望条数（「最新论文」固定 8，「测试检索」固定 3） | `8` |
| `{from}` | 时间范围起点日期 | `2026-10-02` |
| `{to}` | 今天 | `2026-11-01` |
| `{days}` | 时间范围天数（不限时为 3650） | `30` |
| `{year}` | 当前年份 | `2026` |
| `{api_key}` | API Key（**留空时整个参数会被自动剔除**，避免 `api_key=` 空值导致 401） | `abc123` |

补充规则：

- URL 里**没写** `{query}` 时，系统会自动追加 `?query=<关键词>`（或 `&query=...`）；
- 返回值只要**能递归找到**一个「含 `title` 的数组」就行，放在 `results` / `items` / `data` / `papers` / `works` / `message.items` / `results[].bibjson` 里都能识别；
- PDF 直链会自动从 `best_oa_location.pdf_url`（OpenAlex）、`openAccessPdf.url`（Semantic Scholar）、`link[] + content-type`（Crossref）、`link[] + type=fulltext`（DOAJ）等常见位置提取。

### 4. 四个预设的完整示例

| 预设 | API 地址（原样粘贴即可） | 需要 Key？ | PDF 来源 | 备注 |
|---|---|---|---|---|
| **OpenAlex** | `https://api.openalex.org/works?search={query}&filter=from_publication_date:{from}&per-page={limit}` | 否 | `best_oa_location.pdf_url` | 覆盖 2.5 亿+ 文献，免 Key 首选；`search=` 是全字段模糊匹配，想更精准可换成 `filter=title_and_abstract.search:{query}` |
| **Semantic Scholar** | `https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,abstract,venue,year,publicationDate,authors,openAccessPdf,externalIds` | 建议填 | `openAccessPdf.url` | 免 Key 也能用，但共享池限流严重（常见 429），有 Key 更稳；`fields` 参数决定返回哪些字段，删掉 PDF 相关字段就拿不到全文链接；**不支持按日期过滤**（选它时时间范围会自动切到「不限」） |
| **Crossref** | `https://api.crossref.org/works?query.bibliographic={query}&filter=from-pub-date:{from}&sort=published&order=desc&rows={limit}` | 否 | `link[]` 中 `content-type: application/pdf` | DOI 元数据最全（期刊/作者/年份）；出版商全文多为付费墙，取不到时系统会兜底去 OpenAlex / arXiv 找开放版本 |
| **DOAJ** | `https://doaj.org/api/search/articles/{query}?pageSize={limit}` | 否 | `link[]` 中 `type: fulltext` | 只收录**开放获取**期刊，PDF 命中率最高，适合「一定要拿到全文」的场景；**不支持按日期过滤**（选它时时间范围会自动切到「不限」） |

### 5. 手把手例子

**例 A：用 OpenAlex 抓「骨架动作识别」近 30 天最新论文（免 Key，最推荐）**

1. 预设选 `OpenAlex`；名称自动填 `OpenAlex`；
2. 时间范围选 `近 30 天`；
3. 点「测试检索」→ 显示 `✓ 关键词「…」命中 3 条，2 条可直接下载 PDF`；
4. 点「保存」→「添加数据源」弹窗自动关闭，回到「最新论文」面板：右上「自定义API」已自动勾选，
   下拉里的 `OpenAlex` 已自动打勾（带「近30天」标记）。再按例 A 添加 `DOAJ`，两个源就都打勾、会被同时抓取；
5. 点右上「立即刷新」→ 列表出现该站最新论文（卡片前缀 `[OpenAlex]`），有 PDF 的点「导入知识库」即可入库。

**例 B：用 Semantic Scholar 并带 Key**

1. 预设选 `Semantic Scholar`；
2. API Key 填你申请的 Key（`x-api-key` 会自动带上，不用改 URL）；
3. 时间范围随便选（S2 不支持日期过滤，选了也不影响请求）；
4. 「测试检索」通过后「保存」。

**例 C：接入课题组自建服务**

假设师兄写了个 Flask 接口：`GET http://10.0.0.5:5000/search?kw=xxx&n=8`，返回：

```json
{"papers": [{"title": "A Paper", "date": "2026-10-01", "abstract": "...",
             "pdf": "http://10.0.0.5:5000/files/1.pdf", "doi": "10.1234/xyz"}]}
```

填法：

- 名称：`课题组文献库`
- API 地址：`http://10.0.0.5:5000/search?kw={query}&n={limit}`
- API Key：留空（若服务端校验，则填 Key，或用 `?token={api_key}` 写进 URL）
- 系统会识别 `papers` 数组、`title`、`date`、`abstract`、`pdf` 字段。

### 6. 典型错误：填入的是「产品介绍页」

最常见的失败案例：名称填 `Unpaywall`、地址填 `https://unpaywall.org/products/api`，点「测试检索」报
`检索失败`。因为这是 Unpaywall 的**功能说明网页**，返回 HTML 而不是 JSON，所以必然报
`返回内容不是 JSON，请确认填的是 API 地址而不是文档页/首页`。

另外 **Unpaywall 本身也不适合作为「最新论文」数据源**：它的接口是
`https://api.unpaywall.org/v2/{doi}?email=you@example.com`，**按 DOI 精确查单篇的开放获取状态**，
不支持关键词检索，也没有「最新论文列表」。查 OA 全文这件事系统已经内置（见下文 PDF 兜底），
想按方向抓最新论文请用 **OpenAlex / Semantic Scholar / DOAJ**。

判断口诀：**能在浏览器里直接看到 `{` 开头的 JSON、且带 `{query}` 的，才是对的地址。**

### 7. 抓取时 PDF 是怎么拿到的

1. **先看返回字段**：从 JSON 里深度查找 PDF 链接（含 `pdf_url` / `openAccessPdf.url` / `best_oa_location` / `link[]` 等）；
2. **再兜底**：还是没有，则依次按 DOI 查 OpenAlex 开放版本 → 按标题查 OpenAlex → 按标题查 arXiv 预印本。
   兜底**最多 3 条、并发执行、总时长 8 秒封顶**，超时的条目直接标记为「无开放 PDF」，不会拖住整个抓取；
3. **下载时再兜一次**：如果拿到的是文章落地页（HTML），会自动从页面里解析 `citation_pdf_url` / `link[type=application/pdf]` / 首个 `.pdf` 链接，再下第二次；
4. 下载到的内容校验 `%PDF` 文件头，不是真 PDF 就报错，不会把网页当论文塞进知识库。

### 8. 「测试检索」报错对照表

| 提示 | 原因 | 处理 |
|---|---|---|
| `返回内容不是 JSON，请确认填的是 API 地址而不是文档页/首页：<html…>` | 填成了网页/文档页 | 换成真正的 API 地址（见第 4 节预设） |
| `接口可达但没解析出论文列表，请检查返回结构` | 返回了 JSON，但没有「含 title 的数组」 | 检查是否需要 `fields=` 之类参数，或返回结构是单对象而非列表 |
| `HTTP Error 401 / 403` | Key 缺失、无效，或鉴权方式不匹配 | 补 Key；必要时把 `{api_key}` 写进 URL 改走查询参数 |
| `HTTP Error 429` | 触发限流（Semantic Scholar 免 Key 时常见） | 填 Key，或换 OpenAlex / DOAJ |
| `HTTP Error 404` | 路径不对（例如把 Unpaywall 的 `/v2/{doi}` 当成关键词接口） | 对照官方文档核对路径与参数名 |
| `测试接口不存在（HTTP 404）：后端尚未加载 /api/tools/test-source…` | 后端跑的是**旧代码**：改了 `app.py` 但进程没重启 | 关掉原服务窗口（Ctrl+C）再启动；`start.bat` 发现 8000 端口被占用时会**直接开浏览器而不重启**，见 Q10 |
| `Timeout / WinError 10061` | 网络不可达，或 `config.py` 的 `HTTPS_PROXY` 指向没开的代理端口 | 见「常见问题」Q2 / Q2.1 |
| `certificate verify failed` | 校园网/安全软件替换证书 | 保持 `ALLOW_INSECURE_TLS = True`，重启服务 |

### 9. 数据存在哪里

保存后写入 `front_window/fetch_settings.json`：

```json
{
  "keywords": "skeleton-based action recognition",
  "sources": ["arxiv", "pubmed", "custom"],
  "custom_api_url": "",
  "custom_sources": [
    {
      "id": "src_openalex",
      "name": "OpenAlex",
      "url": "https://api.openalex.org/works?search={query}&filter=from_publication_date:{from}&per-page={limit}",
      "api_key": "",
      "doc_url": "https://docs.openalex.org/api-entities/works",
      "days": 30
    },
    {
      "id": "src_doaj",
      "name": "DOAJ",
      "url": "https://doaj.org/api/search/articles/{query}?pageSize={limit}",
      "api_key": "",
      "doc_url": "https://doaj.org/api/v3/docs",
      "days": 0
    }
  ],
  "custom_source_ids": ["src_openalex", "src_doaj"],
  "custom_source_id": "src_openalex"
}
```

- `custom_sources`：多数据源列表，可增删多条；
- `custom_source_ids`：**参与抓取的源 id 列表（可多条，会同时抓取并合并结果）**。下拉里取消勾选某个源，就是从该数组里移除；
- `custom_source_id`：仅保留向后兼容，等于 `custom_source_ids` 的第一条；新版界面不再使用单选；
- `days`：时间范围天数，`0` = 不限，每个源各自独立；
- `sources` 里必须有 `"custom"`，否则即使勾选了数据源也不会参与抓取；
- 老版本只存 `custom_api_url`（单条 URL）时，启动会自动迁移成一条名为「自定义API」的数据源，无需手工改；
- 老版本只有 `custom_source_id`（单选）时，启动会把该字段迁移成 `custom_source_ids`（多选，默认全部启用）。

### 10. 与「标识符导入」「URL 导入」的分工

- **添加数据源**：让系统**按关键词主动去某站点抓最新论文**（拉取列表）；
- **导入菜单 → 标识符导入**：已知 DOI / arXiv ID / PubMed ID，**补全单篇元数据并入库**；
- **导入菜单 → 粘贴 URL**：已知某篇 PDF 的下载地址，**直接下载入库**。

三者互补：先用数据源扫最新，再用标识符/URL 补单篇。

### 11. 按论文方向抓取（以某篇论文为参考）

不只是按固定关键词抓，「领域最新论文」还能**以知识库里的某篇论文为参考，抓同一研究方向的最新论文**。

**用法**

1. 打开「领域最新论文」面板（科研问答页右上角 `🕐 领域最新论文`）；
2. 检索栏最左侧的下拉从 `按关键词` 切到 `按论文方向`，关键词输入框会变成一个论文下拉；
3. 下拉默认选中**最近一次对话涉及的那篇论文**（也可手动换任意一篇）；
4. 点「立即刷新」→ 面板标题变成 `领域最新论文（方向：《论文标题》 · 抓取于 …）`。

**它怎么把标题变成检索词**

| 步骤 | 说明 |
|---|---|
| 取标题 | 从 `front_window/papers.json` 按 `paper_id` 取 `title`（缺省用 `paper_id`） |
| 提炼关键词 | 优先让 LLM 输出 4~8 个英文关键词；离线 Mock / LLM 不可用 / 输出含中文时，退回词法方案：去停用词后取**最长的 3 个实词**（长词多为术语），全大写词降为小写 |
| 示例 | `MULTICOMPLEXES IN ACTION: STRONG FORMALITY OF COMPACT HYPERKÄHLER MANIFOLDS` → `multicomplexes formality hyperkähler` |
| 放宽重试 | 命中不足 3 条时，自动用前两个词再搜一次（标题术语往往过窄，只搜到原文本身）；不再放宽到单个词，避免 `multicomplex` 这类同形异义词带偏 |
| 去自引用 | 参考论文本身不算「相关方向的新论文」，会从结果里剔除（登记簿标题可能被截断，因此按前缀包含判断） |
| 抓不到时 | 面板提示「没有抓到论文（可放宽时间范围、换个数据源，或点「立即刷新」重试）」 |

> 该模式只影响这一次手动刷新，不会改写 `fetch_settings.json` 里的关键词；
> 后台定时抓取仍按保存的关键词跑（每 `FETCH_INTERVAL_HOURS` 小时覆盖一次 `latest_papers.json`）。
> 接口契约：`POST /api/tools/latest/refresh`，body 可选 `{"paper_id": "..."}`。

**怎么确认它真的生效了**

- 下拉切换（或换参考论文）后会**立刻按新模式重抓**，不必再点「立即刷新」；
- 面板标题括号里会写清楚本次用的是什么：`关键词：xxx` 还是 `方向：《论文标题》`；
- 列表顶部会出现两类提示条：`部分来源失败：arxiv: The read operation timed out`（哪个源挂了）、
  `本次未能按论文方向检索…`（后端没重启 / 论文 id 失效，此时列表仍是关键词结果）。

### 12. 怎么确认某个数据源（如 OpenAlex）真的能连上、能抓到论文

**方法一（推荐）：用「测试检索」按钮**

「添加数据源」弹窗 → 选好预设 / 填好 URL 与 Key → 点「测试检索」。它不走缓存、不看来源勾选，
直接用表单内容实发一次请求：

- 成功：`✓ 关键词「…」命中 3 条，2 条可直接下载 PDF`，下面列出前 3 条标题；
- 失败：给出具体原因（非 JSON / 401 / 429 / 超时），对照第 8 节的报错表处理。

OpenAlex 首次实测约 10 秒（末尾会对没有 PDF 的条目并发查开放版本），属正常，不是卡死。

**方法二：看最近一次抓取的记录**

打开 `front_window/latest_papers.json`，看两个字段：

| 字段 | 说明 |
|---|---|
| `sources` | 必须是 `["arxiv","pubmed","custom"]` 才会调用自定义源。**只有 `["arxiv","pubmed"]` 说明「自定义API」没勾选或没保存，OpenAlex 根本没参与** |
| `error` | 出现 `custom[OpenAlex]: ...` 说明该源调用了但失败，冒号后是真实原因（如 `HTTP Error 429`）；多个源同时抓取时，一个源失败只会记它自己，不影响其它源。为 `null` 且没有 `[OpenAlex]` 卡片，说明调用成功但该方向确实没返回结果 |

**方法三：命令行直连自检（绕开界面与配置）**

```powershell
cd e:\比赛\agent
python -c "import scholar_sources as s; ps=s.search_custom('https://api.openalex.org/works?search={query}&filter=from_publication_date:{from}&per-page={limit}','LLM',5,days=30); print('命中', len(ps)); [print(p['published'], p['title'][:60], 'PDF' if p['pdf_url'] else '') for p in ps]"
```

能打印出标题+日期就说明网络与解析都正常。若报 `HTTP Error 429`，是 OpenAlex 公共池限流，等几分钟或换
Semantic Scholar / Crossref 预设。

---

## 七、近期修改（自定义数据源与最新论文链路）

### 1. 前端（`front_window/index.html`）

| 改动 | 说明 |
|---|---|
| 数据源管理改为独立弹窗 | 原「搜索栏内嵌一行 URL 输入」升级为「添加数据源」弹窗：名称 / API 地址 / API Key / 时间范围 / 文档地址 + 已添加列表（可删除） |
| 多数据源 + 多选抓取 | 右侧下拉列出全部数据源，**每行一个勾选框，勾上即参与抓取（可多选、同时抓取并合并去重）**；行内可看文档（「文档」链接）、可删除；勾选变化立即写 `custom_source_ids` 并生效。按钮文字汇总当前状态：`自定义API（2 个）` / `1/3 个源` / `未选择数据源` |
| 站点预设 | OpenAlex / Semantic Scholar / Crossref / DOAJ 一键填充地址与文档链接 |
| 时间范围 | 近 7 / 30 / 90 天 / 近 1 年 / 不限，写入 URL 的 `{from}` |
| 「测试检索」按钮 | 用未保存的表单内容实跑一次，显示命中条数、PDF 条数与前 3 条标题 |
| 失败原因可见 | 失败时显示后端返回的 `error` / `detail`；都没有则显示 `HTTP 状态码`，404 直接提示「后端未重启，需重新启动服务」，不再只显示「检索失败」四个字 |
| 预设时效性 | DOAJ / Semantic Scholar 接口不支持日期过滤，选这两个预设时自动把时间范围设为「不限」，避免界面显示「近30天」却实际不生效 |
| 按论文方向检索 | 检索栏新增「按关键词 / 按论文方向」下拉；切到论文方向时关键词框换成论文下拉，默认选中最近一次对话涉及的论文；切换下拉或换论文即自动按新模式重抓，刷新时把 `paper_id` 发给后端 |
| 关键词即时生效 | 点「立即刷新」时会把**输入框里当前的关键词**一起提交（`{"keywords": "..."}`），改完词不必先点「保存设置」；与已保存关键词不同时会提示「本次按输入框关键词抓取（未保存，定时任务仍用已保存的关键词）」 |
| 来源勾选即生效 | 点「立即刷新」时若发现来源勾选与已保存的不一致（典型：勾了「自定义API」却没点保存），会先静默保存再抓取，避免「勾了却没用上 OpenAlex」 |
| 抓取结果可核对 | 标题括号写明本次是 `关键词：…` 还是 `方向：《…》`；列表顶部提示条显示源失败原因（如 arXiv 超时）与「未能按论文方向检索」 |
| 空结果提示 | 抓到 0 条时直接提示「没有抓到论文（可放宽时间范围、换数据源或重试）」，不再只渲染空列表 |
| 文档链接 | 选中源的 `doc_url` 直达「查看文档」 |
| 切换即生效 | 下拉切换后静默同步到后端并提示，避免刷新时仍用上一条源；面板开着会用新源立即重抓 |
| 添加后自动化 | 自动勾选「自定义API」、选中新源；若面板开着立刻重抓 |
| 来源标签 | 卡片前缀由固定的 `[自定义API]` 改为实际源名，如 `[OpenAlex]` |
| 误填拦截 | URL 域名以 `docs.` / `doc.` 开头时弹窗提醒「像文档站不是 API」 |
| 清理死代码 | 移除旧内嵌输入行的 `lt-custom-url` / `toggleCustomUrl` 等引用，`source-mask` 纳入 Esc 统一关闭 |

### 2. 后端（`scholar_sources.py`）

| 改动 | 说明 |
|---|---|
| 占位符扩展 | `{query}` `{limit}` `{max_results}` `{per_page}` `{from}` `{to}` `{days}` `{year}` `{api_key}` |
| 时间窗 | `search_custom(..., days=N)`，`{from}` = 今天 − N 天；`days=0` 视为不限 |
| 无 `{query}` 兼容 | 自动追加 `query=` 参数；空 `api_key=` 等空参数自动剔除 |
| 列表定位增强 | 优先 `results/items/data/papers/works/docs/…` 等键，递归深度 5；兼容 `message.items`、`results[].bibjson`、`metadata` 外壳 |
| 字段兼容 | OpenAlex 的 `abstract_inverted_index` 还原摘要；Crossref `date-parts` 转 `YYYY-MM-DD` 并补零；S2 的 `externalIds.DOI`；`authorships[].author` 作者名 |
| PDF 深度提取 | `best_oa_location` / `openAccessPdf` / `link[]`(content-type、type=fulltext) / `oa_url` 等嵌套位置 |
| PDF 兜底链 | 无 PDF 时按 DOI → 标题查 OpenAlex，再按标题查 arXiv；**最多 3 条、并发、8 秒封顶** |
| 排序 | 结果按发布时间倒序（最新在前） |
| 来源名 | 新增 `source_name` 参数，返回结果的 `source` 用真实站点名 |

`fetch_latest_papers(paper_id=...)` 新增「按论文方向」链路：标题 → 研究方向检索词（LLM 优先，词法兜底取最长 3 个实词）
→ 命中不足 3 条时用前两词放宽重试 → 剔除参考论文自身；返回值新增 `mode` / `ref_paper` / `ref_paper_id`。

### 3. 其他文件

| 文件 | 改动 |
|---|---|
| `front_window/app.py` | 保存设置时保留并夹紧 `days`（0–3650）；新增 `POST /api/tools/test-source`，用表单内容实测数据源并返回前 3 条与 PDF 命中数 |
| `arxiv_tool.py` | `download_pdf` 增加落地页兜底：返回 HTML 时解析 `citation_pdf_url` / `link[type=application/pdf]` / 首个 `.pdf` 链接再下第二次 |
| `arxiv_tool.py` | `search_arxiv` 新增 `strict` 参数：多词查询拼成 `all:a AND all:b`。原来 `all:skeleton action recognition` 会被 arXiv 当宽松匹配（实测命中 34 万条），再按提交时间排序就全是无关新论文；统一检索（`search_arxiv_unified`）启用该参数，其他调用方行为不变 |
| `arxiv_tool.py` | AND 拼接前剔除 `of / for / the / in …` 等虚词：关键词带 `of` 时会拼出 `all:of`，arXiv 实测直接返回 0 条（`skeleton-based of action segmentation` 由 0 条恢复到 15 条） |
| `front_window/fetch_settings.json` | 修正历史脏数据：原 `custom_api_url` 误填为 `https://docs.openalex.org`（文档站，必报错），改为可用的 OpenAlex 数据源（近 30 天） |
| `start.bat` | 端口 8000 已被占用时补一条提示：旧进程不会加载新代码，改过后端需先关掉原服务窗口；避免「改了代码却访问到旧服务」 |

### 4. 对话接口流式化（SSE，Token 级增量渲染）

| 文件 | 改动 |
|---|---|
| `Agent.py` | `get_llm()` 的 `ChatOpenAI` 开启 `streaming=True`：让节点内 `invoke()` 走流式 API 并 emit token 回调，供 `astream_events` 捕获；`invoke()` 结果仍聚合为完整消息，翻译 / 改写等其它接口行为不变 |
| `front_window/app.py` | `POST /api/chat` 由同步 JSON 改为 `StreamingResponse`（`text/event-stream`）：用 `graph_app.astream_events(version="v2")` 驱动，`on_chat_model_stream` → `token` 事件（按 `langgraph_node` 过滤 `router` 意图分类 token），思考日志 → `log` 事件，流结束 → `final` 事件（权威 `answer` + 富数据），异常 → `error` 事件；HITL 恢复改为第二次 `astream_events(None)` |
| `front_window/index.html` | 发送主流程改用 `fetch` + `ReadableStream` 手写解析 SSE：`token` 逐字累积到回答气泡、`log` 实时追加思维链面板、`final` 渲染图谱 / 图像 / 论文卡片 / 溯源锚点并覆盖权威回答；提取 `addBadge` / `applyResult` 复用；`API_VERSION` 与 `NEED_API_VERSION` 同步 +1（`2026-11-03-5`） |

### 5. 项目工作空间化与管理论文（创建后可调整收录）

| 改动 | 说明 |
|---|---|
| 侧边栏「项目」→「选择工作空间」 | 参考 VS Code 工作空间交互：分组标题改为「选择工作空间」，条目点击即**切换当前工作空间**（进入项目并把问答检索范围限定在其论文内），当前工作空间**高亮**；不再只是跳转到项目页。未采纳参考样式的「打开本地文件夹」——论文统一来自知识库（已入库并向量化），项目以 `paper_ids` 引用而非本地目录，引入本地文件夹与现有架构冲突 |
| 项目卡片新增「管理论文」 | 弹出论文勾选弹窗（标题 / ID 过滤、全选 / 清空、实时计数），把知识库论文收入或移出项目，**解决新建项目后无法再添加论文的问题**；保存写回 `paper_ids` 并刷新「N 篇论文」计数 |
| 当前工作空间联动 | 管理论文弹窗保存时，若该项目正是当前工作空间，问答检索范围（`state.selected`）同步更新；删除当前工作空间时自动取消高亮 |
| 实现位置 | `front_window/index.html`：新增 `enterProject` / `openProjectPapers` / `renderPpickList`，`ppick-mask` 弹窗，`state.activeProjectId` 与 `.sb-proj.on` 高亮样式；Esc 可关闭弹窗 |

### 6. 工作空间菜单两项化与新建引导流程（参考 VS Code 样式）

| 改动 | 说明 |
|---|---|
| 「+」菜单固定两个选项 | 侧边栏「选择工作空间 +」（与项目页「新建项目」共用）弹出两项：**「＋ 新建项目」**与**「📁 打开本地文件夹」**；原「新建并收入已选论文（N 篇）」合并进新流程（创建后的「管理论文」弹窗自动预勾选知识库已勾选论文），不再单独出现 |
| 新建项目一条龙 | 输入名称创建 → 自动设为当前工作空间 → 自动弹「管理论文」勾选知识库文件（预勾选已选论文）→ 保存后自动切到问答页并聚焦输入框，提示「已就绪，可直接在对话框提问」——覆盖「创建 → 加入知识库文件 → 提问」完整链路 |
| 打开本地文件夹 = 系统选文件夹上传 | 点击后**直接弹系统文件夹选择框**（`<input webkitdirectory>`，可选电脑任意磁盘的文件夹），选中后批量上传入库（≥2 个文件走批量接口 + 查重）；上传完成后 `autoCollectToWorkspace` 把**本次新入库的论文自动收入当前工作空间**并同步问答检索范围，形成「新建工作空间 → 打开本地文件夹 → 选盘内文件夹上传 → 直接提问」闭环 |
| 工作空间可删除 | 侧边栏「选择工作空间」条目 **hover 显示 × 删除按钮**（点击确认后删除，不触发条目切换、不删除知识库论文）；项目卡片 hover 的「删除」按钮保留，两处共用 `deleteProject`；删除当前工作空间自动取消高亮 |
| 知识库为空时 | 创建工作空间后不弹管理论文，改为切到知识库并提示「请先上传 PDF 或拖拽导入，入库后再『管理论文』加入」；也可直接走「打开本地文件夹」选文件夹上传（上传后自动收入） |
| 归档与删除明确区分 | 项目页标签「全部」更名「**进行中**」（实际只显示未归档项目，名实一致）；归档是**可逆收纳**——操作后 toast 提示「项目未删除，切到『已归档』标签可查看并恢复」，删除则需 confirm 且不可恢复，两者效果不再混淆 |
| 卡片动作按钮不再溢出 | 项目卡片 hover 动作行（进入 / 管理论文 / 重命名 / 归档 / 删除）允许换行并采用紧凑样式（`flex-wrap` + 小号按钮），五个按钮不再溢出到卡片外 |
| 实现位置 | `front_window/index.html`：`proj-menu` 两项化（`proj-menu-new` / `proj-menu-local`），`createProject` 引导流程（`ppickEnterAfterSave` 标志），`openProjMenu` 简化，`openProjCreate` 预收入已选论文；`uploadFiles` 重构（记录上传前集合 + `autoCollectToWorkspace` 自动收入），侧边栏 `.sb-proj-del` 删除按钮 |

### 7. 标题层级推断与 LaTeX 公式抽取（阅读器增强）

| 改动 | 说明 |
|---|---|
| 标题树结构化（heading 层级） | 解决「只按文本块切分、未构建 heading 层级」：后端 `_build_heading_toc` 在 PDF 无书签时，逐行用 `get_text("dict")` 收集字号/粗体，正文字号取众数，候选标题按「字号去重聚类 ≤3 档」映射一/二/三级，返回 `toc`（含 `level`），前端目录树按层级缩进（`lv1/lv2/lv3`）。有书签的 PDF 仍优先用 `get_toc` |
| LaTeX 公式抽取 | 解决「无公式识别」：后端 `_detect_formulas` 按「文本短 + 数学符号密度 ≥0.10 + 非完整句子」启发式识别公式块，`reader` 接口新增 `formulas` 字段与 chunk 级 `is_formula` 标记 |
| **LaTeX 还原（规则层，2026-11-05 晚）** | `_unicode_to_latex`：Unicode 数学记号 → LaTeX（希腊字母、∑∫√∏、上下标字符合并为 `^{}`/`_{}`、×÷≤≥∈∪→ 等 90+ 映射）；`_fix_latex_spacing` 修复被 PDF 字距拆散的命令（`\ma t hbf` → `\mathbf`，仅命中已知命令表才合并）；命令尾空格策略防粘连。**零依赖、离线、即时**，随 `reader` 直接下发（`latex` 字段，`latex_source="rule"`） |
| **LaTeX 还原（视觉层，可选）** | `POST /api/papers/{id}/formula-latex`：懒加载 **pix2tex / LaTeX-OCR**（`pip install pix2tex`，含 torch CPU 约 1GB；权重走 HF 镜像下载约 100MB）对公式区域图做视觉识别，还原规则层难覆盖的结构；**未安装/失败自动回退规则层**（HTTP 200 + `source="rule"`），`ENABLE_LATEX_OCR=off` 可彻底禁用。实测：首次调用约 18s（含模型加载），驻留后单条 <1s |
| **版面级语义标注** | `_detect_formulas` 为每条公式产出：`eq_no`（右侧编号 `(2.3)` → `2.3`）、`kind`（display 纯符号 / mixed 含文字）、`section`（所属章节，来自标题树最近条目）、`prev`/`next`（版面相邻正文块片段）——即赛题「版面级语义理解」的轻量落地 |
| 前端「∑ 公式」侧栏 | 阅读器顶栏新增「∑ 公式」开关 + 可拖拽公式侧栏：徽章行（编号/类型/章节）+ 原文 + LaTeX（可复制）+ 上下文 + 「AI 精修」按钮（视觉识别中禁用并显示进度，失败显示回退原因）；点击跳页并高亮；无公式/扫描件时给出空态提示 |
| 翻译面板公式不译 | 中英对照翻译中，`is_formula` 的块直接展示原文（等宽字体 + 底纹），不送入 LLM 翻译；`translatePage` 跳过公式块 |
| 实现位置 | 后端 `front_window/app.py`（`_unicode_to_latex` / `_fix_latex_spacing` / `_detect_formulas` / `_get_latex_ocr` / `formula-latex` 端点）；前端 `front_window/index.html`（`buildFormulaList` / `refineFormulaLatex`）；`config.py`（`ENABLE_LATEX_OCR`）；`requirements.txt`（pix2tex 可选段） |

### 8. 表格 / 图表结构化抽取（阅读器增强）

| 改动 | 说明 |
|---|---|
| 表格结构化抽取 | 后端 `_extract_tables_and_figures`：`page.find_tables()` 检出**有边框线**的表格；`_detect_ruleless_tables` 用 `get_drawings()` 水平线聚合识别**三线表**（top/bottom 横线对 + 区域文本按行/列还原），`table.extract()` 还原为二维行列表（`{page, bbox, rows}`），`reader` 接口新增 `tables` 字段 |
| 图表（figure）识别 | 位图：`get_text("dict")` 中 `type=1` 图片块（过滤 <40pt 图标/logo）；**矢量图**：`_detect_vector_figures` 用 `get_drawings()` 聚合矢量图形区域（过滤细线/图标/超大元素，合并重叠矩形）；均用 `_match_caption` 向下匹配 Fig/Figure/Table/表/图 开头的图注 |
| 区域渲染预览 | `_render_region_png` 用 `page.get_pixmap(clip=bbox)` 把表格/图表/公式区域渲染为 PNG base64，前端侧栏以缩略图预览，直观呈现「结构化抽取到了什么」 |
| 前端「表格图表」侧栏 | 阅读器顶栏新增「▦ 表格图表」开关 + 可拖拽侧栏：列出表格（「表格 · 第 x 页 · r 行 × c 列」）与图表（「图表 · 第 x 页」+ 图注 + 缩略图）；点击表格项展开 HTML 表格并跳页高亮，点击图表项跳页高亮图片区域；公式侧栏同步增加缩略图预览 |
| 局限 | 完全无线的纯文本对齐表格仍是启发式边界；扫描件无文本层时表格检测可能为空。公式 LaTeX 还原已于 2026-11-05 晚补齐（见第 7 节） |
| 实现位置 | 后端 `front_window/app.py`（`_extract_tables_and_figures` / `_detect_ruleless_tables` / `_detect_vector_figures` / `_words_to_cells` / `_render_region_png` / `_match_caption`）；前端 `front_window/index.html`（`buildTablesList` / `toggleTableView`、`reader-tables` 侧栏、`.tbl-thumb` 缩略图） |

### 9. 图像放大查看与上下文面板分界线拖拽（问答视图体验）

| 改动 | 说明 |
|---|---|
| 图像放大查看（lightbox） | 点击「结果增强」面板里的 **Matplotlib 图像**（`#viz-img`）以及阅读器**表格 / 图表 / 公式缩略图**（`.tbl-thumb`），全屏遮罩放大预览原图（`imgview-mask`，点任意处或 Esc 关闭）；用 capture 阶段点击委托拦截缩略图点击，**不触发父级**（跳页 / 展开表格）行为 |
| 面板分界线可拖拽 | 问答视图的聊天区与「上下文面板」之间新增**可拖拽分界线**（`ctx-split`，复用阅读器 `.v-split` 样式）：按住拖动调整面板宽度（240px ~ 60% 视窗），聊天区自动占余下空间；分界线随面板显隐联动（`.ctx.show + .ctx-split` 相邻兄弟选择器，无需 JS 同步），窄屏媒体查询下自动隐藏 |
| 实现位置 | `front_window/index.html`：`openImageView` / `imgview-mask` 遮罩 / capture 点击委托；`ctx-split` 分隔条与专用拖拽逻辑；`.ctx{order:2}` / `.ctx-split{order:1}` 布局 |

### 10. 模块六学术写作辅助 + 整篇翻译 + 历史对话（2026-11-04）

| 改动 | 说明 |
|---|---|
| 论文框架生成（WRITE 分支） | 新增 `writing.py`：`generate_paper_framework` 由 idea 生成 Title/Abstract/Introduction/Related Work；`Agent.py` 新增 `write_node`，`app.py` 注入 `reference_catalog`（库内真实文献清单）+ `_enrich_references`（Crossref 标题检索补 DOI 并缓存到 `papers.json`），References 只引用知识库真实文献，LLM 仅在 Related Work 按 `[n]` 引用清单编号（无幻觉） |
| 架构拓扑图脚本（DIAGRAM 分支） | `writing.py:generate_diagram_scripts` 一次性产出 TikZ / Graphviz / Mermaid / Matplotlib 四格式脚本；前端代码块 + 一键复制 |
| CSV 数据可视化增强 | `visualization.py:execute_data_visualization_v2`：表格 + 折线 + 柱状 + 雷达（polar）+ 三维柱状（mplot3d）；`compute_summary` 统计均值/趋势/相关，`writing.py:generate_figure_caption` 用 LLM 提炼自适应学术图注（模板兜底） |
| 中译英学术翻译 | `app.py` 新增 `POST /api/tools/translate-en` + 前端「工具 → 中译英学术翻译」弹窗；`writing.py:translate_zh_to_en` 保留公式/LaTeX/变量/引用编号 |
| 整篇翻译（原位排版 + Markdown 导出） | `app.py` 新增 `POST /api/papers/{id}/full-translate`（`_build_full_translation`）：抽取全文结构（`get_toc`/`_build_heading_toc` 标题层级 + `_detect_formulas` 公式 + `_extract_tables_and_figures` 表格图表）→ 分块翻译（`_translate_full_blocks`）→ 组装原位排版 Markdown（正文/标题/图注译中文，公式/代码/参考文献保留原文，页码锚点）。**表格分级导出**：结构可信 → Markdown 表格；存疑（`_table_looks_real` 检查矩形性/标点碎片/断数字）→ 嵌入区域截图 data URI；三线表启发式 → 注释。前端「⇄ 整篇翻译」预览（折叠 base64）+ 复制 + 下载 `.md` |
| 历史对话存档与答案回补 | 回答与富数据（图表/框架/脚本/锚点）随消息入档，`persistChat` localStorage 配额不足时逐级降级（去图像 → 去富数据）；`restoreChat` 恢复后自动渲染最近带结果回答，带结果气泡可点击回看；旧版仅存提问的记录通过 `app.py` 的 `GET /api/chat/{id}/replay` 从 LangGraph checkpoint 回补缺失回答并回写存档 |
| 前端健壮性修复 | 历史弹窗 HTML/CSS 补回（此前被删、JS 残留导致脚本在 2533 行中断、知识库与「+」按钮失效）；`bindEl` 防御式绑定；`scipy` 依赖补充（引用图谱分支）；前后端 `API_VERSION` 版本戳对齐 |

### 11. 侧边栏折叠重构 / 回收站 / 登录注册（2026-11-05）

**前端（`front_window/index.html`）**

| 改动 | 说明 |
|---|---|
| 侧边栏宽度与拖拽 | 展开 240px、收起 64px；右缘分隔条可拖拽（160–420px，`localStorage` 记忆 `zy_sb_width`），拖动超过阈值自动展开，切换宽度过渡 `width .3s ease` |
| 折叠态专属状态类 | 新增 `sidebar-collapsed`，与 `collapsed` 同步挂载；所有折叠样式用 `.sidebar.sidebar-collapsed` 双类名 + `!important`，彻底覆盖展开态的 padding / 宽高 / 对齐 |
| 折叠态盒模型 | 菜单项统一 40×40px 正方块（`padding:0`、`justify-content:center`、`margin:0 auto`）、图标 18×18px、菜单间距 4px；选中/hover 背景为正方形圆角块，不再出现横向通栏 |
| 折叠态顶部与底部 | 折叠时隐藏面板切换图标，**点击「智」Logo 展开**；底部回收站、头像、设置齿轮同步居中，分隔线占满窄栏内容区 |
| 折叠态 Tooltip | 图标悬停时右侧弹出深色气泡提示（搜索 / 知识库 / Agent / 项目 / 科研问答 / 工具 / 回收站） |
| 无障碍属性 | 面板切换按钮带 `aria-label="Toggle sidebar"` 与实时同步的 `aria-expanded`，便于自动化脚本判定收起/展开状态 |
| 回收站 | 侧边栏「回收站」打开弹窗：删除的项目 / 论文自动入站（`localStorage` 键 `zy_trash`，保留 30 天，超期自动清理）；项目支持**一键恢复**，论文保留删除记录（若删除时保留 PDF 原文件会标注可重新导入）；支持单条彻底删除与清空回收站，均带二次确认 |
| 登录 / 注册入口 | 侧边栏底部为白色胶囊**分段切换器**（选中项深黑实心 + 白字）+ 语言选择按钮（🌐 CN ▾，点击在 CN/EN 间切换） |
| 登录弹窗 | 浅灰页面背景上的白色大圆角卡片（主色 `#7c3aed`）：居中 Logo + 标题「登录你的账号」；输入框浮动标签、左侧 `+86` 区号、聚焦紫色描边 + 光晕；协议勾选（「使用条款」「隐私政策」紫色链接）；紫色主按钮；「还没有账号？ 注册」；「或」分隔线 + 微信 / 机构登录按钮（注册态微信按钮置灰禁用） |
| 注册（两步） | 第一步：手机号 + 短信验证码（「获取验证码」带 60s 倒计时，演示环境验证码经 toast 展示）→「下一步」；第二步：`设置密码` + `确认密码` + 规则提示「6-20 位，且同时包含字母和数字」→「完成注册」，成功自动登录并关闭弹窗 |
| 登录（手机号 + 密码） | 密码框下方右侧「忘记密码？」紫色链接；未注册 / 密码错误分别提示 |
| 找回密码（两步） | 第一步手机号 + 验证码（校验该号是否注册）→ 第二步新密码 + 确认 →「重置密码」，成功后自动登录 |
| 实时校验 | 手机号格式、验证码 6 位、密码规则、两次输入一致性均输入即校验，错误显示在**对应输入框下方**，修改后自动消失；回车可提交 |
| 状态持久化 | 账号表 `localStorage:zy_users`（`{手机号: {type:'phone', pass, regAt}}`，兼容旧版纯字符串密码）、当前登录用户 `zy_user`；点击头像/用户名可退出登录 |

> 实现位置：`front_window/index.html`（折叠：`.sidebar.sidebar-collapsed` 样式块 + `syncSbArrow`；回收站：`loadTrash` / `renderTrash` / `restoreTrash` + `trash-mask` 弹窗；认证：`renderAuth` / `doLogin` / `doVerifyStep` / `doPasswordStep` / `sendAuthCode` + `#auth-body` 动态渲染，事件全部走 `#auth-card` 委托）。

### 12. 引用含金量筛选 / TrueCite（2026-11-05，对应赛题模块四）

| 改动 | 说明 |
|---|---|
| 新增评估模块 | `citation_quality.py`：确定性规则打分，**不依赖 LLM 与网络**（断网可答辩），每条文献输出 0-100 分 + 五个信号分项 + 中文理由 + 机器可读 `flags` |
| 五维评分 | 元数据完整度 25（作者 / 年份 / 出处 / DOI·arXiv）+ 可核验性 20（DOI 格式、arXiv ID、可信学术域名）+ 发表形态 15（期刊会议 15 / 预印本 9 / 未知 3）+ 使用强度 25（正文实际被引 ≥3 次 25、2 次 20、1 次 14、0 次 0；无正文按中性 12）+ 时效性 10（近 3 年 10 / 10 年 7 / 20 年 3） |
| 应付式引用识别 | 重复条目 -30、无题名 -20、题名过短 -12、`et al.` / `unknown` 占位 -15、年份晚于当前年 -10、正文从未引用（`not_cited_in_text`）直接判 0 分区 |
| 档位与阈值 | ≥70 核心、50-69 支撑、30-49 边缘、<30 存疑；阈值可选 40 / 50 / 65（宽松 / 标准 / 严格），低于阈值或重复的条目 `keep=false`，前端**默认不勾选**，实现"自动剔除" |
| 三种输入 | ① `[1]` 编号参考文献列表（自动切分作者/题名/出处/年份/DOI/arXiv，兼容 `et al.`、单作者、Anonymous）；② BibTeX；③ RIS |
| 接口 | `POST /api/tools/truecite`（原 501 预留 → 已实现）：入参 `{reference_text | bib_text | entries, body_text?, threshold?}`，返回 `{ok, summary{total,kept,dropped,avg_kept_score,tiers,has_body}, items[...]}`；空输入 400，无法解析 400 |
| 批量导入联动 | `POST /api/tools/parse-bibfile` 每条附 `quality{score,tier,tier_label,keep,reasons,flags}` 与 `quality_summary`；导入弹窗标题显示「含金量建议保留 M/N」，卡片显示评分徽章 + 首要理由 |
| 前端交互 | 工具页「引用含金量筛选 · TrueCite」弹窗：粘贴/上传 → 分析 → KPI（解析数 / 建议保留 / 建议剔除 / 保留均分）→ 逐条展示分项小条与理由；支持勾选、复制保留条目（按 GB/T 7714 风格重排）、勾选条目批量导入（仅 `pdf_url` 可自动下载） |
| 版本戳 | 前后端 `API_VERSION` 同步为 `2026-11-05-1`，旧进程会提示重启 |

> 实现位置：`citation_quality.py`（`parse_reference_text` / `normalize_entries` / `score_entry` / `assess`）；`front_window/app.py`（`CitationQualityRequest`、`truecite`、`parse_bibfile`）；`front_window/index.html`（`openCiteQ` / `runCiteQ` / `renderCiteQ` / `updateCiteqSel` + `#citeq-mask` 弹窗、`.cq-*` 样式）。

### 13. 配置 Embedding 模型：哈希向量 → 本地语义向量（2026-11-05）

> **当前状态**：项目现已升级为**百炼云 API `text-embedding-v4`（Qwen3-Embedding 系列，1024 维）**，
> 全库 7650 块已重建并实测通过，详见 §14 路线 C 与 **§15 运维须知**；本节保留本地语义模型阶段的改造记录。

**问题**：平台网关只有 `/chat/completions`（实测 `/v1/embeddings`、`/v1/models` 均 404），
此前 RAG 只能用 `_LocalHashEmbeddings`（词元哈希，1536 维）：只有词面重叠能力，
中文提问查英文论文相关度趋近 0，同义改写完全无法命中，`dedup_service` 被迫把判重阈值放宽到 0.65。

| 改动 | 说明 |
|---|---|
| 新增 `embeddings.py` | 统一工厂 + 三种后端：`remote`（OpenAI 兼容 `/embeddings`）/ `local`（fastembed + ONNX Runtime，**无需 Key、无需 torch、可离线**）/ `hash`（兜底）。统一 `embed_query` / `embed_documents`，并暴露 `backend / model_name / dim / is_semantic`；启动时逐级尝试、失败自动回退，**任何环境都不会因模型问题启动失败** |
| 默认本地模型 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（384 维，50+ 语言，中英互查）；`EMBEDDING_LOCAL_MODEL` 可换 `bge-small-zh-v1.5`（512 维）/ `multilingual-e5-large`（1024 维）等 |
| 维度治理 | 集合维度不再硬编码 1536：先探测模型维度再建集合；空库自动按新维度重建；有数据且维度不符时回退哈希保可用，并打印重建命令 |
| 批量嵌入 | 入库由逐段 `embed_query` 改为 `embed_documents` 分批（默认 16/32），本地 ONNX 一次前向多段，导入速度提升约一个量级；远程接口也减少请求数 |
| 向量库重建 | 新增 `reindex_embeddings.py`：备份 `{id,payload}` → 写临时集合 `__rebuild` → 数量核对 → 原子替换正式集合；支持 `--dry-run/--batch/--backend/--db` |
| 语言自适应 | E5 系列自动加 `query:` / `passage:` 前缀；其余模型不加 |
| 判重阈值 | `DEDUP_SIM_THRESHOLD` 可覆盖；语义后端默认 0.88、哈希 0.65（沿用自动判定） |
| 可观测 | 新增 `GET /api/health`：返回 `embedding{backend,model,dim,semantic,label}` + 集合名 + 点位数量；`semantic=false` 即表示仍在哈希兜底 |
| 检索改写 | `_build_search_query` 保留：语义模型已具备跨语言能力，但把口语化长问句压成论文术语关键词仍能提升召回，注释已同步说明 |
| 运维 | `pip install fastembed`（已加入 `requirements.txt`）；模型缓存 `models/` 已 gitignore；断网演示设 `EMBEDDING_ALLOW_DOWNLOAD=false` 只读缓存 |

> 实测（本机 CPU）：中英跨语言余弦 `"如何提升小样本目标检测的精度？" ↔ "How to improve few-shot object detection accuracy?" = 0.71`，
> 无关句 `"今天的天气" ↔ "graph convolutional network" = 0.02`；7650 个文本块重建耗时约 12 分钟（≈10 段/秒）。
>
> 实现位置：`embeddings.py`（`build_embeddings` / `LocalOnnxEmbeddings` / `RemoteEmbeddings` / `describe`）；`Source Tracing.py`（后端选择 + 维度治理 + 批量入库）；`reindex_embeddings.py`；`front_window/app.py`（`/api/health`）；`config.py`（`EMBEDDING_*`）。

### 14. 部署 Qwen3-Embedding 的三种路线（本机 fastembed / GPU vLLM / 百炼云 API）

Qwen3-Embedding 有三档，**先按硬件选档，再谈部署**（8B 不是"下载下来就能跑"的规模）：

| 模型 | 维度 | 权重体积 | 最低配置 | 本项目接法 | 全库重建（7650 块） |
|---|---|---|---|---|---|
| `Qwen3-Embedding-0.6B` | 1024 | ONNX 量化 ~1.1GB | 8GB 内存 / 纯 CPU | `local`（fastembed，免 torch） | 约 10–20 分钟 |
| `Qwen3-Embedding-4B` | 2560 | ~8GB | 12GB+ 显存 | `remote`（vLLM） | 约 3–8 分钟（GPU） |
| `Qwen3-Embedding-8B` | 4096 | ~16GB | 16GB+ 显存（推荐 24GB） | `remote`（vLLM） | 约 5–15 分钟（GPU） |

> 纯 CPU + 8GB 内存跑 8B 需要 GGUF 量化（约 6GB 常驻），加载即接近内存上限，单段推理 1–3 秒，
> 全库重建要 6 小时以上，**不建议作为答辩方案**。

#### 路线 A：本机 CPU 跑 0.6B（推荐先跑通）

fastembed 提供官方 ONNX 量化版 `Qwen/Qwen3-Embedding-0.6B-Q`，无需 torch、无需魔塔即可加载
（模型来自 HuggingFace，可设镜像）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内加速；已能直连 HF 可省略
.\.venv\Scripts\python.exe -c "from fastembed import TextEmbedding; TextEmbedding('Qwen/Qwen3-Embedding-0.6B-Q', cache_dir='models')"
```

`.env` 改为：

```ini
EMBEDDING_BACKEND=local
EMBEDDING_LOCAL_MODEL=Qwen/Qwen3-Embedding-0.6B-Q
EMBEDDING_LOCAL_QUERY_PREFIX=Instruct: Given a web search query, retrieve relevant academic passages\nQuery:
EMBEDDING_BATCH_SIZE=8
```

#### 路线 B：GPU 机器上跑 8B（vLLM），本机远程调用

**1) 在 GPU 机器上从魔塔下载**

```powershell
pip install modelscope
modelscope download --model Qwen/Qwen3-Embedding-8B --local_dir /models/Qwen3-Embedding-8B
# 只下 safetensors 主体（省流量）：加 --include "*.json" "*.safetensors" "*.py"
```

**2) 起 OpenAI 兼容的向量服务（vLLM ≥ 0.9）**

```bash
pip install "vllm>=0.9.0"
vllm serve /models/Qwen3-Embedding-8B \
  --served-model-name Qwen3-Embedding-8B \
  --task embed --port 8001 \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.90

# 旧版 vLLM 用：--runner pooling（不支持 --task embed 时）
# 自检：
curl http://127.0.0.1:8001/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-8B","input":["hello"]}' | head -c 200
```

**3) 本机 `.env` 指向它**（`EMBEDDING_BASE_URL` 必须写到 `/v1`；模型名与 `--served-model-name` 完全一致）

```ini
EMBEDDING_BACKEND=remote
EMBEDDING_MODEL=Qwen3-Embedding-8B
EMBEDDING_BASE_URL=http://192.168.1.50:8001/v1
EMBEDDING_API_KEY=EMPTY
EMBEDDING_QUERY_PREFIX=Instruct: Given a web search query, retrieve relevant academic passages\nQuery:
EMBEDDING_BATCH_SIZE=16
EMBEDDING_TIMEOUT=180
```

> 若 vLLM 只能跑在本机但无 GPU 直通，也可用 WSL2 / Docker（`--gpus all`），端口映射到 8001 即可。
> 若用 llama.cpp（`llama-server -m ...gguf --embeddings --pooling last -c 8192 --port 8001`），配置完全一样。

#### 路线 C：阿里云百炼 API 直连 Qwen3-Embedding（✅ **2026-11-05 已实测打通并切换生效**）

百炼上 Qwen3-Embedding 系列的 **API 模型 ID 是 `text-embedding-v4`**（官方注明"属于 Qwen3-Embedding 系列"，
没有名为 `Qwen3-Embedding-8B` 的接口 ID）。OpenAI 兼容端点为
`https://dashscope.aliyuncs.com/compatible-mode/v1`。

**开通与取 Key**：打开 [bailian.console.aliyun.com](https://bailian.console.aliyun.com) → 开通百炼模型服务
（需实名）→ 「API-KEY」页创建（`sk-` 开头）。新用户有 **100 万 Token 免费额度**（开通后 90 天内有效），
超出后 ¥0.5/百万 Token 上下——本项目 7650 个文本块约 250 万 Token，全量重建一次成本约 ¥1。

**`.env` 配置（三行即可切换）**：

```ini
EMBEDDING_BACKEND=remote
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=sk-你的百炼Key
EMBEDDING_QUERY_PREFIX=Instruct: Given a web search query, retrieve relevant academic passages\nQuery:
EMBEDDING_BATCH_SIZE=10
EMBEDDING_TIMEOUT=60
EMBEDDING_DIMENSIONS=0
```

**百炼特有的两条硬限制（代码已适配，配置别改错）**

| 限制 | 值 | 应对 |
|---|---|---|
| 单批条数 | **最多 10 条**（一次请求 ≤10 个文本，单条 ≤8192 Token） | `EMBEDDING_BATCH_SIZE` 保持 ≤10，调大必报 400 |
| 输出维度 | 2048 / 1536 / **1024（默认）** / 768 / 512 / 256 / 128 / 64 | `EMBEDDING_DIMENSIONS=0` 用默认 1024；想更强可设 2048（重建库） |
| 限流 | 免费/低等级账号 QPS 较低，重建时 429 常见 | 代码已内置指数退避重试（2/5/10/20/40s），遇到 429 自动等，无需干预 |

配好后先严格自检再重建（见下文「切换后必做」）。重启后 `GET /api/health` 应显示
`backend=remote, model=text-embedding-v4, dim=1024`。

**✅ 2026-11-05 实测记录（本机已按此路线切换生效）**

| 项目 | 结果 |
|---|---|
| 自检 | `python embeddings.py remote` → `backend=remote, model=text-embedding-v4, dim=1024`；自检相似度「骨架动作识别 ↔ skeleton-based action recognition」= **0.857**（本地 MiniLM 为 0.781） |
| 全库重建 | 7650 块，`--batch 10`，**耗时 278 秒（≈28 段/秒）**，未触发一次 429；数量核对通过、原子替换成功，备份 `reindex_backup_20261105_181821.jsonl` |
| 中文问英文库 · 推荐分支 | 「骨架动作识别在小样本场景下有哪些工作？」→ 命中 One-shot Skeleton Action Recognition / Meta-Baseline 等 5 篇高对口论文 |
| 中文问英文库 · 溯源分支 | 点名问 PRISM → 最高锚点 **0.79**（标题页），页码准确，回答与方法原文一致 |
| 语义泛化 | 「训练时没见过的类别上怎么表现好？」（措辞与论文用词完全不同）→ 仍精准命中 PRISM 第 2/1 页（0.55–0.59） |

> 其他服务商同理：`EMBEDDING_BASE_URL` 换成其端点（如硅基流动 `https://api.siliconflow.cn/v1`）、
> `EMBEDDING_MODEL` 填其模型 ID（如 `BAAI/bge-m3`）即可，无需任何本地文件。

#### 切换后必做：重建向量库

维度变了（384 → 1024 / 4096），旧向量必须整库重算：

```powershell
# 1) 停止 Web 服务（Qdrant 本地模式是单进程文件锁）
# 2) 严格自检：连不上会直接报错，不会静默回退哈希
.\.venv\Scripts\python.exe embeddings.py remote
# 3) 重建（先 dry-run 看规模）
.\.venv\Scripts\python.exe reindex_embeddings.py --dry-run
.\.venv\Scripts\python.exe reindex_embeddings.py --batch 16
# 4) 启动服务，确认 /api/health 的 embedding.backend/model/dim 与预期一致
```

存储估算：4096 维 × 7650 块 ≈ **125MB** 向量（Qdrant 本地库会翻倍占用，磁盘留 1GB 足够）。
判重阈值建议随模型调整：换 Qwen3 系列后同文相似度普遍更高，可从 `DEDUP_SIM_THRESHOLD=0.90` 起调。

**踩坑清单**

- `EMBEDDING_BASE_URL` 少写 `/v1` → 404；`EMBEDDING_MODEL` 与 `--served-model-name` 不一致 → 404。
- **百炼报 `AccessDenied.Unpurchased`**：Key 有效但**模型服务未开通**（新建 Key 后必须先在控制台
  开通「百炼大模型服务」并实名认证，否则所有模型都拒绝）→ 去模型广场开通 `text-embedding-v4` 即可。
- `.env` 改了却不生效：多半是**编辑器没保存**（`EMBEDDING_API_KEY=` 等号后仍为空），用
  `python -c "import config; print(bool(config.EMBEDDING_API_KEY))"` 验证是否真的读到了。
- 8B 首次加载 30–120 秒，`EMBEDDING_TIMEOUT` 需调大，否则探测阶段就超时。
- Qwen3 官方要求**只在查询侧**加 `Instruct: ...\nQuery:`（本项目已实现：`embed_query` 加、入库不加），
  两侧都加反而掉点。
- 远程服务临时挂掉时 `auto` 会回退到 local 的 384 维模型，与 1024/4096 维库不匹配；生产环境建议
  固定 `EMBEDDING_BACKEND=remote`，让故障显式暴露，而不是静默降级。

### 15. 当前生效的 Embedding 配置与运维须知（2026-11-05 起）

**当前生效配置**（`.env`，Key 已填、已实测）：

```ini
EMBEDDING_BACKEND=remote
EMBEDDING_MODEL=text-embedding-v4          # 百炼 Qwen3-Embedding 系列
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_BATCH_SIZE=10                    # 百炼单批上限，勿调大
EMBEDDING_DIMENSIONS=0                     # 默认 1024 维
```

库状态：`academic_papers` 集合 **7650 点 / 1024 维**（`GET /api/health` 可随时核对）。

**日常使用不需要做任何事**：新上传的论文入库时自动用 `text-embedding-v4` 生成向量，
与库内 1024 维向量一致；只有**换模型或改维度**才需要重建（见不变式）。

**四条不变式（违反必出问题）**

| 操作 | 是否需要重建向量库 |
|---|---|
| 只改 `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_TIMEOUT` / `EMBEDDING_BATCH_SIZE` | ❌ 不需要，重启服务即可 |
| 换 `EMBEDDING_MODEL`（不同家族，如 bge-m3）或改 `EMBEDDING_DIMENSIONS` | ✅ 必须：停服务 → `reindex_embeddings.py` |
| 改 `EMBEDDING_BACKEND`（remote ↔ local） | ✅ 两边维度不同（1024 vs 384），必须重建到目标后端 |
| 删除/更换百炼 API Key | ❌ 不重建，但重启服务后自检会失败，直到填入有效 Key |

**费用与配额**

- 免费额度 **100 万 Token**（开通后 90 天内）；本项目全库重建一次约 **250 万 Token ≈ ¥1**，日常问答检索几乎不花钱。
- 用量在百炼控制台「费用中心 / 模型观测」可查；429 限流代码已自动退避重试（2/5/10/20/40s），日志可见。
- 额度耗尽会报 `Arrearage`/`AllocationQuota` 类错误 → 充值或换回本地后端（需重建）。

**⚠️ 断网答辩预案（重要）**

百炼是云 API。答辩现场若不可达 `dashscope.aliyuncs.com`，回退链为：
remote 失败 → 本地 MiniLM（384 维）→ 与 1024 维库不匹配 → **回退哈希向量**（检索严重退化，但仍能跑通流程）。
三选一预案：

1. **推荐：确保现场网络可达**。演示前跑一次自检即可确认：
   `python embeddings.py remote`（打印 `text-embedding-v4 / 1024 维` 即正常）。
2. **彻底离线预案**：提前把 `EMBEDDING_BACKEND=local` + MiniLM 384 维重建一版库（约 12 分钟），
   断网时把 `.env` 切回 local 直接可用——但切回 remote 又要再重建一次，只在确定断网时才做。
3. **最低保障**：什么都不做，断网时系统自动回退哈希向量，流程能跑、`/api/health` 会显示
   `semantic=false`，适合「只演示功能不比检索质量」的场合。

**Key 安全**

- Key 在 `.env`（已 gitignore），不会进仓库；**不要**把 Key 粘贴到聊天记录、截图或提交到代码里。
- 若怀疑泄露：百炼控制台删除旧 Key → 创建新 Key → 改 `.env` 一行 → 重启服务，向量库无需任何变动。

---

## 八、常见问题

**Q1：RAG 提示「Embedding 语义模型不可用」？**
当前默认走百炼 `text-embedding-v4`（云 API）。出现回退提示时按顺序排查：
1. **百炼报 `AccessDenied.Unpurchased`**：模型服务没开通——去百炼控制台开通「百炼大模型服务」（需实名）；
2. **报 `invalid_api_key` / Missing credentials**：`.env` 里 `EMBEDDING_API_KEY=` 没填或没保存，用
   `python -c "import config; print(bool(config.EMBEDDING_API_KEY))"` 验证；
3. **本地后端（`EMBEDDING_BACKEND=local`）不可用**：`pip install fastembed`（含 onnxruntime）；
   首次运行需联网下载模型（约 240MB）到 `models/`；断网演示时保持 `EMBEDDING_ALLOW_DOWNLOAD=false` 只读缓存；
4. 实在无法加载时系统仍会用哈希向量跑通全流程（`GET /api/health` 里 `embedding.semantic=false` 即为此状态）。
另见 §15「断网答辩预案」。

**Q2：arXiv / GitHub 调用失败？**
当前网络需能访问这些学术站点（国内可直连，速度偏慢）。若需代理，在 `config.py` 填 `HTTPS_PROXY`（如 `http://127.0.0.1:7890`）。代理仅作用于学术 API，不影响 LLM 接口。

**Q2.1：点「立即刷新」报 `[WinError 10061] 目标计算机积极拒绝`？**
说明 `config.py` 里的 `HTTPS_PROXY` 指向了一个**没有程序监听**的本地端口（典型：填了 `127.0.0.1:7897` 但 Clash 没开）。二选一：把代理软件打开并核对端口；或把 `HTTPS_PROXY` 置空改为直连。

**Q2.2：报 `certificate verify failed` / 抓不到数据？**
多为校园网或安全软件用自签证书解密 HTTPS，Python 默认不信任。把 `config.py` 的 `ALLOW_INSECURE_TLS` 保持为 `True`（默认），程序会自动降级重试一次并打印一条提示；如需严格校验再改为 `False`。改完配置**必须重启服务**才生效。

**Q3：终端中文乱码？**
PowerShell 执行 `chcp 65001`，或只看前端页面（编码正常）。

**Q4：改了端口怎么访问？**
端口由启动命令决定：`uvicorn app:app --port 8080` → 访问 `http://localhost:8080`。局域网访问加 `--host 0.0.0.0`。

**Q5：换 Embedding 模型后检索异常 / 提示维度不匹配？**
不要清库重导。先停止服务（Qdrant 本地模式是单进程文件锁），执行：
```bash
python reindex_embeddings.py --dry-run   # 先看规模：旧库 1024 维 -> 新模型维度
python reindex_embeddings.py --batch 10  # 备份 -> 重算 -> 临时集合核对 -> 原子替换
```
脚本会把 `{id, payload}` 备份成 JSONL，重建失败时正式集合保持原样。完成后重启服务即可。

**Q6：qdrant 提示「已被其他实例占用」？**
本地 Qdrant 同一时刻只允许一个进程打开，请只保留一个服务进程（或改为 Qdrant Server 模式）。

**Q7：「添加数据源」里 API 地址该填什么？测试检索报 `返回内容不是 JSON`？**
要填**返回 JSON 的接口地址**（带 `{query}` 占位符），不是网站首页或 API 文档页。报这个错基本就是填成了网页，
例如 `https://unpaywall.org/products/api`。直接抄第六节第 4 条的预设地址即可。逐字段说明见
**第六节「自定义数据源（「添加数据源」）填写指南」**。

**Q8：勾了「自定义API」也加好了数据源，但「最新论文」里没有该站结果？**
按顺序检查：① `fetch_settings.json` 里 `sources` 是否含 `"custom"`（不勾选不会抓）；② 右侧下拉里该源是否打了勾
（`custom_source_ids` 里有没有它的 id —— 现在的下拉是多选，可以多个源同时抓）；③ 关键词是否过于具体 + 时间范围过窄
（如「近 7 天」），可先点「测试检索」换成「不限」再试；④ 上一条源的报错会显示在面板顶部，形如
`custom[DOAJ]: HTTP Error 429`，点「立即刷新」可复现。

**Q9：为什么有的论文卡片没有「导入知识库」按钮？**
说明该条目没拿到可下载的 PDF 直链（闭源期刊且 arXiv 无预印本）。系统已自动做过兜底（DOI/标题查 OpenAlex、
按标题查 arXiv，最多 3 条、8 秒封顶），仍没有就只显示「无开放 PDF」。可换成 DOAJ 这类开放获取源，
或用「导入菜单 → 粘贴 URL / 标识符导入」手动补。

**Q10：改了代码，页面上新功能还是旧样子 / 新按钮提示 404？**
后端进程不会自动加载新代码。`start.bat` 检测到 8000 端口已被占用时会**直接打开浏览器、不会重启服务**，
所以改完 `app.py` 后必须：① 在原来的服务黑窗口按 `Ctrl+C`（或直接关掉窗口）；② 重新运行 `start.bat`；
③ 浏览器按 `Ctrl+F5` 强制刷新（`index.html` 有前端缓存）。
想省掉手工重启可在启动命令后加 `--reload`：`uvicorn app:app --port 8000 --reload`。

**Q10.1：如何重启 `start.bat`？**

`start.bat` 本身**不会自动重启**：它检测到 8000 端口已被占用时，会直接打开浏览器然后退出（脚本第 40–48 行），旧进程仍用旧代码。要真正重启必须「先停旧服务 → 再重跑 → 强刷页面」，三步：

**① 停掉旧服务**（任选其一）：

- 方式 A：找到标题为 `智研 Agent 一键启动` 的黑窗口，按 `Ctrl+C`（或直接关闭窗口）；
- 方式 B：命令行结束占用 8000 的进程：

```powershell
netstat -ano | findstr :8000     # 记下 LISTENING 那行最后的 PID
taskkill /PID <PID> /F
```

**② 重新运行**：

```powershell
cd e:\比赛\agent
.\start.bat
```

（或直接双击 `start.bat`）

**③ 浏览器强刷**：`index.html` 有前端缓存，按 `Ctrl+F5` 强制刷新。

> 为什么不建议用 `--reload` 热重载：项目启动入口特意没开 `--reload`（`front_window/app.py` 末尾注释说明），
> 因为 reload 会二次导入模块、导致后台定时抓取线程被重复启动。改代码后请按上面三步手工重启。

**Q11：最新论文是不是写死的？为什么切了「按关键词 / 按论文方向」，抓到的还是那几篇？**
列表不是写死的：它来自 `front_window/latest_papers.json` 缓存，缓存由实时抓取写入（`time` 字段就是抓取时间）。
两者一样通常是下面四种原因，按顺序排查：

1. **后端没重启**（最常见）。打开 `front_window/latest_papers.json` 看有没有 `"mode"` 字段：
   没有 = 旧代码写的，旧代码不认 `paper_id`，选什么模式都走关键词。重启服务即可（见 Q10）。
2. **看标题括号**：写 `关键词：…` 说明本次确实按关键词跑的；写 `方向：《…》` 才是按论文方向。
   若选了论文方向却仍显示关键词，列表顶部会有一条提示说明原因。
3. **只读了缓存没重抓**：打开面板（`🕐 领域最新论文`）默认只读缓存。切换下拉会自动重抓，
   也可点右上「立即刷新」。
4. **来源太少导致看起来一样**：`fetch_settings.json` 里 `sources` 若只勾了 PubMed（或 arXiv 超时），
   同一关键词下 PubMed 返回的总是那几篇。列表顶部的提示条会告诉你哪个源失败了。
   要抓某站点最新论文，记得勾选「自定义API」并选好数据源。
5. **改了关键词但结果没变**：看标题括号里的 `关键词：…` 是不是你刚输入的那个。不是的话，
   说明后端用的是「已保存的关键词」（旧版本必须点「保存设置」才生效；新版本点「立即刷新」会直接提交输入框内容）。
   另外关键词里带 `of / for / the` 这类虚词时，旧逻辑会拼出 `all:of` 导致 0 结果，新版本已自动剔除。

---

## 九、前端页面结构与后端预留接口

### 1. 页面结构（`front_window/index.html` 单文件）

```
左侧导航（可收起：展开 240px / 收起 64px，可拖拽调宽）
├── 品牌区：智研学术 + 面板图标收起按钮（折叠后按钮隐藏，点击「智」Logo 展开）
├── 搜索        → 全局搜索浮层（Ctrl+K，按标题定位论文）
├── 知识库      → 视图 view-library：表格 + 搜索 + 上传下拉 + 拖拽导入
├── Agent       → 视图 view-agent ：任务输入框 + 模式胶囊 + 案例回放卡片墙
├── 项目        → 视图 view-project：项目卡片 + 存储用量 + 排序 / 归档
├── 科研问答    → 视图 view-qa    ：对话流 + 上下文面板（日志 / 图谱 / 锚点）
├── 工具        → 视图 view-tools ：TrueCite / 插图转 PPT / 图表转 Excel / 数据转图
├── 选择工作空间 → 侧边栏工作空间切换（前 6 个未归档项目，点击即切换，当前项高亮）
└── 回收站 + 用户区
    ├── 回收站（已删除项目 / 论文，30 天可恢复）
    └── 未登录：登录 / 注册分段切换器 + 语言选择；已登录：头像 + 用户名 + 设置（点击用户名退出登录）
```

弹窗层：PDF 阅读器、版本 Diff、领域最新论文、标识符导入、RIS/BibTeX 批量导入、论文多选、全局搜索、回收站、登录 / 注册 / 找回密码（两步式）。

### 2. 对话接口新增参数（`POST /api/chat`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `paper_ids` | `list[str]` | 前端「已选 N 项」显式限定检索论文；多篇时逐篇检索后综合对比（优先于消息文本匹配） |
| `thinking_level` | `str` | 「思考强度」：`light` / `medium` / `deep`，映射溯源 `top_k` = 3 / 5 / 8（默认 `medium`，与历史行为一致） |

#### 流式返回协议（SSE，`text/event-stream`）

`POST /api/chat` 以 **SSE** 流式返回（前端用 `fetch` + `ReadableStream` 逐块解析；`EventSource` 仅支持 GET、无法携带 POST body，故不用）。每个事件形如 `data: <json>\n\n`，按 JSON 的 `type` 字段区分：

| `type` | 触发时机 | 主要字段 | 前端处理 |
|---|---|---|---|
| `log` | 预处理 / 路由分类 / HITL 打断与恢复 / 分支完成，逐条推送 | `content` | 实时追加到右侧「思维链」面板 |
| `token` | LLM 正文逐 token 增量（仅 RAG / PLAN 分支，其余分支为确定性文本） | `content` | 逐字累积到回答气泡 |
| `final` | 流结束，一次携带权威结果与富数据 | `session_id` `branch` `answer` `log` `paper_id` + 分支富数据（`graph` / `image`+`caption` / `tool_papers` / `anchors`+`paper_ids`） | 覆盖回答（补全溯源后缀）、渲染图谱 / 图像 / 论文卡片 / 溯源锚点 |
| `error` | 工作流执行异常 | `message` | 显示错误提示 |

实现要点：

- 后端用 `graph_app.astream_events(..., version="v2")` 驱动工作流，捕获 `on_chat_model_stream` 事件产出 token；`router` 的意图分类 token 按 `metadata.langgraph_node` 过滤，避免混入正文；
- `ChatOpenAI` 已开启 `streaming=True`（`get_llm()`），使节点内 `invoke()` 也逐 token 回调；离线 Mock 模式（未配置 `OPENAI_API_KEY`）无 token 事件，前端退化为 `final` 一次性返回；
- GRAPH 分支的 HITL 打断在流式下保留：第一次 `astream_events` 在 `graph_node` 前暂停 → 推日志 → 自动确认 → 第二次 `astream_events(None)` 恢复；
- 富数据（图谱 / 图像 / 论文卡片 / 溯源锚点）非流式内容，统一由 `final` 事件返回。

### 3. 预留接口（前端已调用，后端实现后前端零改动生效）

未实现时统一返回 `501` + `{"reserved": true, "message": "..."}`，前端按下列策略降级：

| 接口 | 契约 | 前端当前降级行为 |
|---|---|---|
| `GET /api/projects` | `{"projects":[{"id","name","paper_ids","created","updated","archived"}]}` | 项目存 `localStorage`（`zy_projects`） |
| `POST /api/projects` | 入参 `{"projects":[...]}` → `{"ok":true}` | 静默忽略（仍本地保存） |
| `POST /api/search` | 入参 `{"query","top_k"}` → `{"hits":[{"paper_id","page_num","text","score"}]}` | 搜索浮层用本地标题过滤 |
| `POST /api/tools/truecite` | 入参 `{"bib_text"}` → `{"items":[{"cite","valid","reason"}]}` | 工具卡片走现有 RIS/BibTeX 解析导入 |
| `POST /api/tools/figure-to-ppt` | 入参 `{"paper_id","page_num","bbox"}` → `{"ok","task_id","download_url"}` | Toast 提示「接口已预留」 |
| `POST /api/tools/figure-to-excel` | 入参同上 → `{"ok","task_id","download_url","rows"}` | Toast 提示「接口已预留」 |
| `POST /api/papers/export` | 入参 `{"paper_ids","format"}` → `{"ok","download_url","count"}` | Toast 提示「接口已预留」（知识库「传输」按钮） |
| `GET /api/trash` | `{"items":[{"type","id","deleted_at"}]}` | **前端已本地实现**（`localStorage:zy_trash`，保留 30 天，项目可恢复）；后端接入后可改为服务端统一存储 |
| `GET /api/papers/columns` | `{"columns":[{"key","label","type"}]}` | Toast 提示「接口已预留」（知识库「添加列」） |

> 补充实现时只需在 `front_window/app.py` 中把对应函数体的 `_reserved(...)` 替换为真实逻辑并返回上表契约，前端无需改代码。

---

## 十、模块与赛事评分点对照

| 模块 | 文件 | 对应评分点 |
|---|---|---|
| 多 Agent 路由 + HITL | `Agent.py` | Agent 技术 35%（Task Planning / Function Calling） |
| 复合任务编排（PLAN） | `Agent.py` | 多步推理与任务分解 |
| 带锚点 RAG 溯源 + 行级精准高亮 | `Source Tracing.py`、`index.html` | 抗幻觉 Source Tracing（UX 8%） |
| 引用图谱 + 真实引用边 + 基石挖掘 | `graph and generation.py` | 场景创新 25%（模块五） |
| Future Work / 研究空白分析 | `Agent.py`（PLAN）、`graph and generation.py` | 研究空白自动分析 |
| 语义去重与版本演进 | `dedup_service.py`、`version_dedup.py` | 论文去重与演进分析 |
| CSV 智能绘图（表格/折线/柱状/三维雷达 + 自适应图注） | `visualization.py` | 模块六 学术 Copilot（数据可视化） |
| 学术写作辅助（论文框架 + 无幻觉 References + 拓扑图脚本 + 中译英） | `writing.py`、`Agent.py`（WRITE/DIAGRAM）、`app.py` | 模块六 学术 Copilot（写作辅助） |
| 整篇翻译（原位排版 + Markdown 导出） | `app.py`（`/full-translate`）、`index.html` | 模块三 中英对比翻译 |
| 历史对话存档与答案回补 | `app.py`（`/replay`）、`index.html` | 交互体验 / 产品完成度 |
| 自主工具调用与闭环探索 | `arxiv_tool.py`、`scholar_sources.py`、`bibparse.py` | Tool Use & Environment Interaction |
| 双向联动阅读（跳页高亮 + 划词提问） | `index.html` | 交互体验 / 产品完成度 |
| 任务式 Agent 首页（模式胶囊 + 案例回放） | `index.html`（view-agent） | 产品完成度 / 易用性 |
| 知识库表格化管理 + 多选限定检索 | `index.html`（view-library） | 交互体验 |
| 科研项目组织与存储用量 | `index.html`（view-project） | 场景创新（课题级工作流） |
| 科研工具集（含 3 个预留能力） | `index.html`（view-tools） | 学术 Copilot 工具链 |
