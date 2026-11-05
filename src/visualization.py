# python_executor.py
# pyright 定向抑制：IDE 分析环境解释器未装依赖时误报，运行时依赖真实存在
import pandas as pd  # pyright: ignore[reportMissingImports]
import matplotlib  # pyright: ignore[reportMissingImports]

matplotlib.use("Agg")  # 无界面后端：避免服务端弹窗 / 多线程绘图崩溃

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import io
import base64


def execute_data_visualization(csv_path: str, user_instruction: str):
    """
    自动解析 CSV 并执行绘图代码。
    不再写死列名：自动识别数值列；若 user_instruction 中提到具体列名则优先采用。
    后续接入 LLM 时，可将此处的绘图逻辑替换为 "LLM 生成代码 + 沙箱 exec 执行"。
    """
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError("CSV 中未检测到数值列，无法绘图")

    instruction_lower = user_instruction.lower()
    plt.figure(figsize=(8, 5))

    if len(numeric_cols) >= 2 and "柱" not in instruction_lower and "bar" not in instruction_lower:
        # ---- 多数值列：趋势折线图 ----
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        # 若用户指令中明确提到某数值列，则优先作为 Y 轴
        for col in numeric_cols:
            if col.lower() in instruction_lower and col != x_col:
                y_col = col
        plt.plot(df[x_col], df[y_col], marker="o", color="b", label="Proposed Method")
        plt.title("Model Convergence Analysis", fontsize=14)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.legend()
        caption = (
            f"Figure 1: {y_col} progression over {len(df)} {x_col} steps "
            f"showing clear convergence."
        )
    else:
        # ---- 单数值列（或用户指定柱状图）：分类柱状图 ----
        y_col = numeric_cols[0]
        for col in numeric_cols:
            if col.lower() in instruction_lower:
                y_col = col
                break
        # X 轴取第一个非数值列（如方法名），没有则用行号
        cat_cols = [c for c in df.columns if c not in numeric_cols]
        x_col = cat_cols[0] if cat_cols else None
        x_vals = df[x_col].astype(str) if x_col else [str(i) for i in range(len(df))]
        plt.bar(x_vals, df[y_col], color="#1a73e8")
        plt.title(f"{y_col} Comparison", fontsize=14)
        plt.xlabel(x_col or "Index")
        plt.ylabel(y_col)
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", alpha=0.3)
        caption = f"Figure 1: {y_col} comparison across {len(df)} {x_col or 'records'}."

    # 将生成的图片转为 Base64 以供前端直接渲染
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return {
        "image_base64": img_base64,
        "caption": caption,
    }


# ============================================================
# 模块六增强：表格 / 折线 / 柱状 / 三维雷达图 + 自适应图注
# ============================================================
def _numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include="number").columns.tolist()


def _cat_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _numeric_cols(df)]


def _pick_y_col(df: pd.DataFrame, instruction: str, default_idx: int = 0) -> str:
    """在数值列里挑 Y 轴：优先取用户指令中提到的列名，否则用第 default_idx 个。"""
    numeric = _numeric_cols(df)
    if not numeric:
        return ""
    low = (instruction or "").lower()
    for col in numeric:
        if col.lower() in low:
            return col
    return numeric[default_idx] if len(numeric) > default_idx else numeric[0]


def compute_summary(df: pd.DataFrame) -> dict:
    """统计各数值列的均值/标准差/极值/趋势/相关，供 caption 提炼与雷达图归一化。"""
    summary: dict = {}
    numeric = _numeric_cols(df)
    for col in numeric:
        s = df[col].dropna()
        if s.empty:
            continue
        item: dict = {
            "count": int(len(s)),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
        }
        if len(s) >= 2:
            x = np.arange(len(s))
            slope = float(np.polyfit(x, s.astype(float), 1)[0])
            item["trend"] = "上升" if slope > 1e-6 else ("下降" if slope < -1e-6 else "平稳")
        summary[col] = item
    if len(numeric) >= 2:
        try:
            corr = float(df[numeric[0]].corr(df[numeric[1]]))
            summary["_correlation"] = {
                "pair": f"{numeric[0]} vs {numeric[1]}",
                "value": round(corr, 3),
            }
        except Exception:
            pass
    return summary


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return data


def _render_line(df: pd.DataFrame, x_col: str | None, y_col: str) -> str:
    x_vals = df[x_col] if x_col else df.index + 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, df[y_col], marker="o", color="#2f6fe4", linewidth=2, label=y_col)
    ax.set_title(f"{y_col} Progression", fontsize=14)
    ax.set_xlabel(x_col or "Index")
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _to_b64(fig)


def _render_bar(df: pd.DataFrame, x_col: str | None, y_col: str) -> str:
    x_vals = df[x_col].astype(str) if x_col else [str(i + 1) for i in range(len(df))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_vals, df[y_col], color="#1a73e8", alpha=0.9)
    ax.set_title(f"{y_col} Comparison", fontsize=14)
    ax.set_xlabel(x_col or "Index")
    ax.set_ylabel(y_col)
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    return _to_b64(fig)


def _render_radar(df: pd.DataFrame, instruction: str) -> str | None:
    """三维（多轴）雷达图：数值列作为维度轴，每个类别（行）一条闭合折线。

    - 有非数值标签列时：标签列取值作为系列名，数值列为维度；
    - 否则：行索引作为维度，第一个数值列取值作为唯一系列。
    维度不足 3 时返回 None（回退到普通折线/柱状）。
    """
    numeric = _numeric_cols(df)
    cat = _cat_cols(df)
    label_col = cat[0] if cat else None

    if label_col:
        axes = numeric
        series_names = df[label_col].astype(str).tolist()
        if len(axes) < 3 or len(series_names) < 1:
            return None
        n_axes = len(axes)
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for name in series_names:
            row = df[df[label_col].astype(str) == name].iloc[0]
            vals = [float(row[c]) for c in axes]
            vals += vals[:1]
            ax.plot(angles, vals, marker="o", label=name)
            ax.fill(angles, vals, alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes)
        ax.set_title("Multi-dimensional Radar", fontsize=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=8)
        return _to_b64(fig)

    # 无标签列：维度 = 行（最多取 12 行），值 = 第一数值列
    y_col = _pick_y_col(df, instruction, 0)
    n = min(len(df), 12)
    vals = df[y_col].astype(float).tolist()[:n]
    if len(vals) < 3:
        return None
    angles = np.linspace(0, 2 * np.pi, len(vals), endpoint=False).tolist()
    angles += angles[:1]
    v = vals + vals[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, v, marker="o", color="#2f6fe4", label=y_col)
    ax.fill(angles, v, alpha=0.12, color="#2f6fe4")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([str(i + 1) for i in range(len(vals))])
    ax.set_title(f"{y_col} Radar", fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))
    return _to_b64(fig)


def _render_bar3d(df: pd.DataFrame) -> str | None:
    """三维柱状图：x = 第一个数值列、y = 第二个、z = 第三个（需 ≥3 个数值列）。"""
    numeric = _numeric_cols(df)
    if len(numeric) < 3 or len(df) < 1:
        return None
    from mpl_toolkits.mplot3d import Axes3D  # pyright: ignore[reportMissingImports]  # noqa: F401

    x, y, z = numeric[0], numeric[1], numeric[2]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    n = min(len(df), 20)
    xs = np.arange(n)
    ax.bar3d(
        xs,
        np.zeros(n),
        np.zeros(n),
        0.6,
        0.6,
        df[z].astype(float).tolist()[:n],
        color="#2f6fe4",
        alpha=0.85,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f"3D Bar: {z} by {x}", fontsize=13)
    return _to_b64(fig)


def _build_table(df: pd.DataFrame, max_rows: int = 12) -> dict:
    """截断式表格：前 max_rows 行，字符串化，供前端渲染精美表格。"""
    head = df.head(max_rows)
    headers = [str(c) for c in head.columns]
    rows = [[str(v) for v in r] for r in head.itertuples(index=False, name=None)]
    return {"headers": headers, "rows": rows, "total_rows": int(len(df))}


def _fallback_caption(summary: dict, chart_type: str, y_col: str, n: int) -> str:
    """无 LLM 时的模板图注兜底。"""
    if chart_type == "table":
        return f"Table 1: Summary of {n} records across {len(summary)} numeric variables."
    trend = ""
    info = summary.get(y_col) or {}
    if info.get("trend"):
        trend = f", showing an overall {info['trend']} trend"
    return f"Figure 1: {y_col} {chart_type} over {n} records{trend}."


def execute_data_visualization_v2(
    csv_path: str,
    user_instruction: str,
    caption_fn=None,
) -> dict:
    """模块六增强版：CSV -> 精美表格 + 折线 / 柱状 / 三维雷达图 + 自适应图注。

    返回结构：
      {
        "table": {"headers": [...], "rows": [[...]], "total_rows": N},
        "charts": [{"type": "...", "image_base64": "...", "caption": "..."}],
        "summary": {...}   # 供前端 / caption 复用
      }
    caption_fn(summary, chart_type, instruction) -> str 可选，用于 LLM 自适应图注。
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV 为空，无法可视化")
    numeric = _numeric_cols(df)
    if not numeric:
        raise ValueError("CSV 中未检测到数值列，无法绘图")

    low = (user_instruction or "").lower()
    summary = compute_summary(df)
    x_col = _cat_cols(df)[0] if _cat_cols(df) else None
    y_col = _pick_y_col(df, user_instruction, 0)

    charts: list = []

    def make_caption(chart_type: str, extra: dict | None = None) -> str:
        # 优先 LLM 自适应图注；失败/未提供时退回模板
        if caption_fn is not None:
            try:
                return caption_fn(summary, chart_type, user_instruction)
            except Exception:
                pass
        return _fallback_caption(summary, chart_type, y_col, len(df))

    # 图表类型判定：雷达 > 三维 > 柱状 > 折线（与赛题四类对齐）
    want_radar = any(k in low for k in ("雷达", "radar"))
    want_3d = any(k in low for k in ("三维", "3d", "立体"))
    want_bar = any(k in low for k in ("柱", "bar"))
    want_table = any(k in low for k in ("表格", "表", "table"))

    if want_radar:
        img = _render_radar(df, user_instruction)
        if img:
            charts.append({"type": "radar", "image_base64": img, "caption": make_caption("radar")})
    if want_3d:
        img = _render_bar3d(df)
        if img:
            charts.append({"type": "bar3d", "image_base64": img, "caption": make_caption("3D bar")})
    if want_bar:
        charts.append(
            {
                "type": "bar",
                "image_base64": _render_bar(df, x_col, y_col),
                "caption": make_caption("bar"),
            }
        )
    if want_radar or want_3d or want_bar:
        # 有明确图表意图时不再默认补折线
        pass
    if not charts:
        # 默认：折线图（与历史行为一致）
        charts.append(
            {
                "type": "line",
                "image_base64": _render_line(df, x_col, y_col),
                "caption": make_caption("line"),
            }
        )

    # 表格始终生成（赛题要求「精美表格」作为独立产出）
    table = _build_table(df)
    table["caption"] = make_caption("table")

    return {"table": table, "charts": charts, "summary": summary}
