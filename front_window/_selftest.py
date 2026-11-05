# _selftest.py — 前端 API 自测脚本（验证后可删除）
import os
from fastapi.testclient import TestClient
from app import app, AGENT_DIR

c = TestClient(app)
print("GET / ->", c.get("/").status_code)

# ---- 1. 上传真实 PDF 入库 ----
pdf_path = os.path.join(AGENT_DIR, "研究生比赛系统方案推荐 .pdf")
with open(pdf_path, "rb") as f:
    r = c.post("/api/upload", files={"file": ("研究生比赛系统方案推荐.pdf", f, "application/pdf")})
print("upload ->", r.status_code, r.json())

# ---- 2. 知识库列表 ----
r = c.get("/api/papers")
print("papers ->", [(p["paper_id"], p["chunks"]) for p in r.json()["papers"]])

# ---- 3. 入库后的 RAG 溯源问答（应返回锚点而非拒答） ----
r = c.post("/api/chat", json={"message": "评分标准中 Agent 技术占多少比例？"})
d = r.json()
print("RAG branch ->", d["branch"])
print("answer ->", d["answer"][:100])
print("anchors ->", len(d.get("anchors", [])), "条")
if d.get("anchors"):
    a = d["anchors"][0]
    print("首条锚点 -> 第", a["page_num"], "页, 相似度", round(a["score"], 3))

# ---- 4. 删除论文 ----
pid = r = c.get("/api/papers").json()["papers"][0]["paper_id"]
r = c.delete(f"/api/papers/{pid}")
print("delete ->", r.status_code, r.json())
print("剩余论文 ->", len(c.get("/api/papers").json()["papers"]))
