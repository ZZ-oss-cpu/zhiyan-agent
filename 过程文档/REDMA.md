# 变更记录（Changelog）

本文件记录项目最近的改动，按日期倒序排列。详细运行方式见 `README.md`。

---

## 2026-10-31　启动方式重构与依赖修复

### 背景

项目此前的唯一启动方式是 `cd front_window` 后执行 `uvicorn app:app --port 8000`：

- 忘记切目录、或在错误的目录执行，`Source Tracing.py` 的相对路径 `./qdrant_db` 会指向别处，
  表现为「知识库突然变成 0 篇」；
- `app.py` 没有 `if __name__ == "__main__"` 入口，`python app.py` 无法启动服务；
- 项目里存在一个指向 `D:\Python`（Python 3.8.6，已不存在）的空壳 `.venv`，
  IDE 自动选中它后所有第三方库都提示「无法解析导入」；
- `requirements.txt` 缺少 Web 端依赖，新建虚拟环境后仍无法启动。

### 改动清单

| 文件 | 改动 | 原因 |
|---|---|---|
| `Source Tracing.py` | 第 10—13 行：`import fitz` 改为 `try: import pymupdf as fitz / except ImportError: import fitz`，两行均加 `# pyright: ignore[reportMissingImports]` | 消除 PyMuPDF 2.x 的 `fitz` 弃用警告；兼容旧版本仅有 `fitz` 顶层名的环境；静态检查环境未索引依赖时不误报 |
| `front_window/app.py` | 第 35—38 行：新增 `if __name__ == "__main__": os.chdir(BASE_DIR)` | 以脚本方式运行时先把工作目录切到 `front_window`，保证 Qdrant 相对路径 `./qdrant_db` 与 uvicorn 模式一致 |
| `front_window/app.py` | 第 1159—1168 行：新增 `if __name__ == "__main__": uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")` | 使 `python app.py` 可直接启动服务；不启用 `reload`，避免二次导入导致定时抓取线程重复启动 |
| `start.bat` | **新建**（项目根目录，约 70 行） | 一键启动：双击即用 |
| `requirements.txt` | 末尾新增 Web 依赖：`fastapi>=0.110`、`uvicorn>=0.29`、`python-multipart>=0.0.9` | 原清单只覆盖 RAG / 图谱 / 绘图，缺 Web 三件套，新建虚拟环境后服务起不来 |
| 环境 | 删除失效的 `.venv`（`pyvenv.cfg` 指向 `D:\Python`、版本 3.8.6、内部无任何包），在 `E:\比赛-综合\agent\.venv` 重建为 **Python 3.14.5**（`home = C:\Python314`） | 修复 IDE「无法解析导入 pymupdf」；统一解释器版本 |

### `start.bat` 行为说明

1. `chcp 65001` 切 UTF-8，`cd /d "%~dp0"` 定位到脚本所在目录（支持中文路径）。
2. 选择解释器：优先 `.venv\Scripts\python.exe`，但会先试导入 `fastapi, uvicorn, qdrant_client, pymupdf`；
   不通过则回退到 PATH 中的 `python`；两者都不可用时给出可复制的安装命令并退出。
3. 检测 8000 端口：已在监听则直接打开浏览器（避免重复启动触发 Qdrant 单进程锁）。
4. 启动 `uvicorn app:app --host 127.0.0.1 --port 8000`，并用隐藏的 PowerShell 延迟 4 秒打开
   `http://127.0.0.1:8000`，避免浏览器比服务先就绪。

### 当前启动方式

| 方式 | 命令 |
|---|---|
| 双击 | `start.bat` |
| 命令行 | `python front_window\app.py` |
| 原有方式（仍可用） | `cd front_window` → `uvicorn app:app --port 8000` |

### 验证状态

- `python -m py_compile front_window/app.py` 通过；IDE lint 无新增告警。
- 未完成端到端启动验证（命令被执行环境转入后台，未取得进程输出）。首次使用请观察窗口输出：
  出现 `Uvicorn running on http://127.0.0.1:8000` 即为成功。

### 遗留事项

- `E:\比赛-综合\agent\.venv` 目前为**空环境**（只有解释器，未装依赖）。
  `start.bat` 会自动回退到全局 `C:\Python314`，不影响使用；若要固定用虚拟环境，执行：

  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 本地 Qdrant 为嵌入式单进程独占：服务运行时不要再并行执行 `python Agent.py` 等自检脚本，
  否则会报「已被其他实例占用」。

---

## 2026-10-30　作品介绍文档生成

### 新增文件

| 文件 | 说明 |
|---|---|
| `智研Agent_作品介绍.docx` | 作品介绍文档，约 590 KB |

### 文档结构

- 封面（作品信息表）、摘要与关键词、目录（Word 域，打开后 `Ctrl+A` → `F9` 更新页码）
- 第 1 章 项目详情及使用方式：五层架构（图 1）、模块清单（表 1）、五分支工作流（图 2 / 表 2）、
  安装步骤、操作路径（表 3）、命令行自检入口
- 第 2 章 项目亮点：9 项亮点总览（表 4）与分节说明，含行级溯源链路（图 3）、语义去重与版本演进（图 4）
- 第 3 章 AI 测试技术落地：测试策略、功能矩阵（表 5）、检索质量对照（表 6）、
  缺陷定位与修复（表 7）、降级与异常测试（表 8）、结论与限制
- 第 4 章 关键技术：技术栈（表 9）、8 个专题小节、关键参数与取值依据（表 10）
- 第 5 章 项目图片：示意图 2 张（图 5、图 6）+ 界面截图清单（表 11）+ 3 个截图占位框（图 7—图 9）
- 第 6 章 发表与获奖：登记表（表 12），信息待作者据实填写
- 第 7 章 项目基础与优化说明：来源、借鉴/自研对照（表 13）、8 项迭代优化、定位差异（表 14）
- 附录 A—D：接口清单（表 A1，20 个路由）、配置项（表 B1）、目录结构、排障速查（表 D1）

### 待补充内容

1. 封面：参赛单位、团队成员、指导教师（现为下划线占位）。
2. 第 6 章表 12：论文 / 专利 / 软著 / 竞赛获奖的实际情况；暂无请填「暂无」。
3. 第 5 章图 7—图 9：知识库列表、RAG 溯源回答、阅读器行级高亮三张界面截图，替换灰色占位框即可。

### 生成方式

文档由 Python 脚本 `python-docx` 生成，配套 4 张示意图由 Matplotlib 绘制（中文字体 Microsoft YaHei）。
生成脚本与中间图片为一次性产物，已清理；如需修改文档，重新生成后覆盖即可。

---

## 回滚提示

上述代码改动均为增量，可单独还原：

- `Source Tracing.py`：将第 10—13 行还原为 `import fitz`。
- `front_window/app.py`：删除第 35—38 行与第 1159—1168 行两个 `if __name__ == "__main__"` 块。
- `requirements.txt`：删除 Web 前端三行依赖。
- `start.bat`：直接删除，不影响原有 uvicorn 启动方式。
