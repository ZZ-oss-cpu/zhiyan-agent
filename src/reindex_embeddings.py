# reindex_embeddings.py
# ============================================================
# 向量库重建工具：切换 Embedding 模型（维度可能变化）后，把已入库论文重新向量化。
#
# 为什么需要它：
#   Qdrant 集合的向量维度在创建时固定。旧库是 1536 维哈希向量，换成 384 维
#   多语言语义模型后，旧向量既维度不符、语义也无效，必须整库重算。
#
# 用法（**先停止 Web 服务**，Qdrant 本地模式是单进程文件锁）：
#   python src/reindex_embeddings.py                 # 按 config.py 当前配置重建
#   python src/reindex_embeddings.py --dry-run       # 只统计，不写入
#   python src/reindex_embeddings.py --db front_window/qdrant_db --collection academic_papers
#   python src/reindex_embeddings.py --backend local --batch 32
#
# 安全措施：
#   1. 先把所有 {id, payload} 备份成 JSONL（默认写入库目录，可用 --backup 指定）；
#   2. 新向量先写入临时集合 __rebuild，数量核对通过后才替换正式集合；
#   3. 任何一步失败，正式集合保持原样（临时集合可手动删除）。
# ============================================================
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import config
from embeddings import build_embeddings, describe

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:  # pragma: no cover
    print("缺少 qdrant-client，请先 pip install qdrant-client")
    sys.exit(1)


def _default_db() -> str:
    """默认库路径：优先 front_window/qdrant_db（app.py 的启动目录），否则 ./qdrant_db。"""
    fw = os.path.join(config.ROOT_DIR, "front_window", "qdrant_db")
    if os.path.isdir(fw):
        return fw
    return os.path.join(config.ROOT_DIR, "qdrant_db")


def _scroll_all(client: QdrantClient, collection: str, with_vectors: bool = False):
    """分页读取集合全部点位。"""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        for p in points:
            yield p
        if offset is None:
            break


def _create(client: QdrantClient, name: str, dim: int):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def _upsert_batches(client, name, points, batch, log_every=200):
    total = len(points)
    for start in range(0, total, batch):
        client.upsert(collection_name=name, points=points[start : start + batch])
        done = min(start + batch, total)
        if done % log_every == 0 or done == total:
            print(f"  写入 {done}/{total}", flush=True)


def _swap_by_rename(db: str, coll: str, tmp: str):
    """把临时集合切换为正式集合（先关闭客户端再改文件，避免本地模式内存残留）。

    为什么不直接用 delete_collection + create_collection：
    qdrant-client 本地模式在同一进程里删除再创建同名集合时，内存中的向量数组仍是旧维度，
    继续写入会得到"新旧维度混排"的损坏集合（实测 1536/384 混排后 DB 无法加载）。
    改为文件系统级替换 + 改写 meta.json：等价于原子换库，且免去二次搬运 7000+ 向量。
    """
    meta_path = os.path.join(db, "meta.json")
    coll_root = os.path.join(db, "collection")
    final_dir = os.path.join(coll_root, coll)
    tmp_dir = os.path.join(coll_root, tmp)
    backup_dir = final_dir + ".bak"

    if os.path.isdir(backup_dir):
        shutil.rmtree(backup_dir)
    if os.path.isdir(final_dir):
        shutil.move(final_dir, backup_dir)  # 保留旧集合目录，切换成功后再删
    if not os.path.isdir(tmp_dir):
        raise RuntimeError(f"临时集合目录不存在：{tmp_dir}")
    shutil.move(tmp_dir, final_dir)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["collections"][coll] = meta["collections"].pop(tmp)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    shutil.rmtree(backup_dir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="按当前 Embedding 模型重建 Qdrant 向量库")
    ap.add_argument(
        "--db", default="", help="Qdrant 本地库路径（默认自动探测 front_window/qdrant_db）"
    )
    ap.add_argument("--collection", default=config.QDRANT_COLLECTION, help="集合名")
    ap.add_argument(
        "--backend", default="", help="auto/remote/local/hash（默认取 config.EMBEDDING_BACKEND）"
    )
    ap.add_argument(
        "--batch", type=int, default=max(8, config.EMBEDDING_BATCH_SIZE), help="批量嵌入/写入大小"
    )
    ap.add_argument("--backup", default="", help="备份 JSONL 路径（默认写入库目录）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = ap.parse_args()

    db = args.db or _default_db()
    if not os.path.isdir(db):
        print(f"[错误] 向量库路径不存在：{db}")
        return 2

    print(f"[1/5] 连接向量库：{db}")
    try:
        client = QdrantClient(path=db)
    except Exception as e:  # 大多是服务未停止导致文件锁被占用
        print(f"[错误] 打开向量库失败：{e}")
        print("       如果 Web 服务正在运行，请先停止（Qdrant 本地模式是单进程文件锁）。")
        return 2

    coll = args.collection
    if not client.collection_exists(coll):
        print(f"[错误] 集合不存在：{coll}")
        return 2

    info = client.get_collection(coll)
    old_dim = int(getattr(info.config.params.vectors, "size", 0) or 0)
    old_count = client.count(coll, exact=True).count

    print(f"[2/5] 当前集合 {coll}：{old_count} 个点位，{old_dim} 维")
    if old_count == 0:
        print("      集合为空，无需重建。")
        return 0

    # strict=True：远程/本地语义模型不可用时直接失败，
    # 绝不允许静默回退 1536 维哈希向量把整个库写坏。
    try:
        emb = build_embeddings(backend=args.backend or None, strict=True)
    except Exception as e:  # noqa: BLE001
        print(f"[错误] 目标 Embedding 后端不可用：{e}")
        print("       请先确认自建服务已启动、EMBEDDING_BASE_URL / EMBEDDING_MODEL 配置正确。")
        return 2
    meta = describe(emb)
    print(f"[3/5] 目标模型：{meta['label']} / {meta['model']}（{meta['dim']} 维）")

    if args.dry_run:
        print(f"[dry-run] 将重建 {old_count} 个点位：{old_dim} 维 -> {meta['dim']} 维")
        return 0

    # ---- 读取 + 备份 ----
    backup_path = args.backup or os.path.join(
        db, f"reindex_backup_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    payloads: list[dict] = []
    ids: list[str] = []
    with open(backup_path, "w", encoding="utf-8") as f:
        for p in _scroll_all(client, coll):
            ids.append(str(p.id))
            payloads.append(p.payload or {})
            f.write(json.dumps({"id": str(p.id), "payload": p.payload}, ensure_ascii=False) + "\n")
    print(f"[4/5] 已备份 {len(payloads)} 条到 {backup_path}")

    texts = [str(pl.get("text") or "") for pl in payloads]
    valid = [(i, pl, t) for i, pl, t in zip(ids, payloads, texts) if t.strip()]
    if not valid:
        print("[错误] 所有点位都没有 text 字段，无法重建。")
        return 2
    if len(valid) < len(payloads):
        print(f"      跳过 {len(payloads) - len(valid)} 个无文本点位")

    # ---- 重新向量化（先写临时集合，核对后再替换）----
    # 临时集合名带上维度：同名 delete+create 在本地模式下会复用旧维度数组，
    # 维度不同就会写坏集合；带维度可保证「删除-重建」永远同维，安全。
    tmp = f"{coll}__rebuild_{int(meta['dim'])}"
    print(f"[5/5] 重新向量化并写入临时集合 {tmp} ...")
    _create(client, tmp, int(meta["dim"]))

    batch = max(1, args.batch)
    t0 = time.time()
    written = 0
    for start in range(0, len(valid), batch):
        group = valid[start : start + batch]
        vectors = emb.embed_documents([t for _, _, t in group])
        points = [
            PointStruct(id=pid, vector=vec, payload=pl) for (pid, pl, _), vec in zip(group, vectors)
        ]
        client.upsert(collection_name=tmp, points=points)
        written += len(points)
        if written % 400 == 0 or written == len(valid):
            speed = written / max(time.time() - t0, 1e-6)
            print(f"  {written}/{len(valid)}（{speed:.0f} 段/秒）", flush=True)

    tmp_count = client.count(tmp, exact=True).count
    if tmp_count != len(valid):
        print(f"[错误] 临时集合点位数量不符（{tmp_count} != {len(valid)}），正式集合未改动。")
        return 3

    # ---- 替换正式集合（关客户端 -> 目录级替换 + 改 meta -> 重开校验）----
    print("      核对通过，替换正式集合 ...")
    client.close()
    try:
        _swap_by_rename(db, coll, tmp)
    except Exception as e:
        print(f"[错误] 集合切换失败：{e}")
        print(f"       临时集合目录仍保留在 collection/{tmp}，可用备份 JSONL 手工恢复。")
        return 4

    client = QdrantClient(path=db)
    final_dim = int(getattr(client.get_collection(coll).config.params.vectors, "size", 0) or 0)
    final_count = client.count(coll, exact=True).count
    client.close()
    print(
        f"\n[完成] {coll}：{final_count} 个点位，{old_dim} 维 -> {final_dim} 维"
        f"（{meta['label']}），耗时 {time.time() - t0:.1f}s"
    )
    print(f"       备份保留在：{backup_path}")
    print("       重新启动服务即可生效。")
    return 0 if final_count == len(valid) else 3


if __name__ == "__main__":
    raise SystemExit(main())
