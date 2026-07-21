from __future__ import annotations

from io import BytesIO
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from analysis import AnalysisResult, build_month_full_vs_daytime, build_std_tables, estimate_peak_cut, summarize_demand_ranges

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


WEEKDAY_JP_ORDER = ["月", "火", "水", "木", "金", "土", "日"]
THEME_PALETTES: Dict[str, list[str]] = {
    "Plotly標準": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"],
    "Colorblind Safe": ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#CC79A7", "#999999"],
    "Bright": ["#ff595e", "#ff924c", "#ffca3a", "#8ac926", "#52a675", "#1982c4", "#4267ac", "#6a4c93", "#f15bb5", "#00bbf9"],
    "Pastel": ["#a8dadc", "#f1faee", "#f4a261", "#e9c46a", "#b8c0ff", "#c9ada7", "#84a59d", "#d4a373", "#b5e48c", "#90dbf4"],
}


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _plotly_png(fig: go.Figure) -> Optional[bytes]:
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def _pillow_line_png(y_values: list[float], title: str, y_label: str = "kW") -> bytes:
    w, h = 1200, 520
    left, top, right, bottom = 80, 50, 30, 80
    chart_w = w - left - right
    chart_h = h - top - bottom

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((12, 10), title, fill="#222222")
    draw.line((left, top, left, h - bottom), fill="#222222", width=2)
    draw.line((left, h - bottom, w - right, h - bottom), fill="#222222", width=2)
    draw.text((8, top), y_label, fill="#444444")

    vals = [float(v) for v in y_values if np.isfinite(v)]
    if vals:
        ymin, ymax = min(vals), max(vals)
        if ymax <= ymin:
            ymax = ymin + 1.0
        points = []
        n = len(y_values)
        for i, y in enumerate(y_values):
            yy = float(y) if np.isfinite(y) else ymin
            x = left if n <= 1 else left + int(i * chart_w / (n - 1))
            yn = (yy - ymin) / (ymax - ymin)
            ypx = h - bottom - int(yn * chart_h)
            points.append((x, ypx))
        if len(points) >= 2:
            draw.line(points, fill="#1f77b4", width=3)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="#1f77b4")

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _matplotlib_line_png(x_values: list[str], y_values: list[float], title: str, y_label: str = "kW") -> bytes:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_values, y_values, marker="o", linewidth=1.6, markersize=2.5)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=70)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _matplotlib_multi_line_png(series: Dict[str, list[float]], title: str, y_label: str = "kW") -> bytes:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, vals in series.items():
        ax.plot(range(len(vals)), vals, linewidth=1.0, alpha=0.8, label=name)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    if len(series) <= 15:
        ax.legend(loc="upper right", fontsize=7)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _matplotlib_bar_png(x_values: list[str], y_values: list[float], title: str, y_label: str = "回数") -> bytes:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_values, y_values, color="#4e79a7")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=70)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _matplotlib_dual_png(x_values: list[str], line_values: list[float], bar_values: list[float], title: str) -> bytes:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(x_values, line_values, color="#e15759", marker="o", linewidth=2.8, markersize=4.5, label="最高デマンド値")
    ax2.bar(x_values, bar_values, color="#59a14f", alpha=0.28, width=0.55, label="最高値記録回数")
    ax1.set_ylabel("最高デマンド値(kW)")
    ax2.set_ylabel("回数")
    ax1.set_title(title)
    ax1.grid(alpha=0.25)
    ax1.tick_params(axis="x", rotation=70)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _fallback_line_png(x_values: list[str], y_values: list[float], title: str, y_label: str = "kW") -> bytes:
    if MATPLOTLIB_AVAILABLE:
        return _matplotlib_line_png(x_values, y_values, title, y_label)
    return _pillow_line_png(y_values, title, y_label)


def _fallback_multi_line_png(series: Dict[str, list[float]], title: str, y_label: str = "kW") -> bytes:
    if MATPLOTLIB_AVAILABLE:
        return _matplotlib_multi_line_png(series, title, y_label)

    merged: list[float] = []
    for vals in series.values():
        merged.extend(vals)
    return _pillow_line_png(merged if merged else [0.0], title, y_label)


def _fallback_bar_png(x_values: list[str], y_values: list[float], title: str, y_label: str = "回数") -> bytes:
    if MATPLOTLIB_AVAILABLE:
        return _matplotlib_bar_png(x_values, y_values, title, y_label)
    return _pillow_line_png(y_values, title, y_label)


def _fallback_dual_png(x_values: list[str], line_values: list[float], bar_values: list[float], title: str) -> bytes:
    if MATPLOTLIB_AVAILABLE:
        return _matplotlib_dual_png(x_values, line_values, bar_values, title)
    return _pillow_line_png(line_values, title)


def _download_png(fig: go.Figure, label: str, file_name: str, key: str, fallback_png: Optional[bytes]) -> None:
    png = _plotly_png(fig)
    if png is None:
        png = fallback_png
    if png is None:
        st.caption("PNG生成に失敗しました。")
        return
    st.download_button(label, data=png, file_name=file_name, mime="image/png", key=key)


def _download_csv(df: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    st.download_button(label, data=_to_csv_bytes(df), file_name=file_name, mime="text/csv", key=key)


def _line_theme() -> str:
    theme = str(st.session_state.get("line_theme", "Plotly標準"))
    return theme if theme in THEME_PALETTES else "Plotly標準"


def _line_width() -> float:
    try:
        return float(st.session_state.get("line_width", 2.5))
    except Exception:
        return 2.5


def _series_color(idx: int) -> str:
    palette = THEME_PALETTES[_line_theme()]
    return palette[idx % len(palette)]


def _matplotlib_table_heatmap_png(df: pd.DataFrame, title: str, text_mode: str) -> bytes:
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    numeric = df.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    rows, cols = values.shape
    fig_w = max(8.0, min(0.6 * cols + 2.5, 24.0))
    fig_h = max(3.0, min(0.45 * rows + 2.5, 20.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.array([0.0])
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    vmed = float(np.nanmedian(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    cmap = LinearSegmentedColormap.from_list("bwr_custom", ["#2c7bb6", "#ffffff", "#d7191c"])
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vmed, vmax=vmax)
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([str(c) for c in numeric.columns], rotation=80, fontsize=8)
    ax.set_yticklabels([str(i) for i in numeric.index], fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("時刻")
    ax.set_ylabel("区分")

    for r in range(rows):
        for c in range(cols):
            val = values[r, c]
            if not np.isfinite(val):
                txt = ""
            else:
                txt = f"{val:.2f}"
            rr, gg, bb = _three_color_rgb(float(val) if np.isfinite(val) else np.nan, vmin, vmed, vmax)
            tcol = _text_color_for_mode(rr, gg, bb, text_mode)
            ax.text(c, r, txt, ha="center", va="center", fontsize=7, color=tcol)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _pillow_table_heatmap_png(df: pd.DataFrame, title: str, text_mode: str) -> bytes:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    rows, cols = numeric.shape
    cell_w, cell_h = 90, 28
    left, top = 140, 80
    w = max(900, left + cols * cell_w + 30)
    h = max(300, top + rows * cell_h + 40)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    draw.text((10, 10), title, fill="#222222")
    for ci, col in enumerate(numeric.columns):
        draw.text((left + ci * cell_w + 4, top - 22), str(col), fill="#333333")
    for ri, idx in enumerate(numeric.index):
        draw.text((8, top + ri * cell_h + 6), str(idx), fill="#333333")

    vals = numeric.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        finite = np.array([0.0])
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    vmed = float(np.nanmedian(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    for ri in range(rows):
        for ci in range(cols):
            val = vals[ri, ci]
            r, g, b = _three_color_rgb(float(val) if np.isfinite(val) else np.nan, vmin, vmed, vmax)
            x0 = left + ci * cell_w
            y0 = top + ri * cell_h
            x1 = x0 + cell_w - 1
            y1 = y0 + cell_h - 1
            draw.rectangle((x0, y0, x1, y1), fill=(r, g, b), outline="#dddddd")
            txt = "" if not np.isfinite(val) else f"{val:.2f}"
            tcol = _text_color_for_mode(r, g, b, text_mode)
            draw.text((x0 + 6, y0 + 7), txt, fill=tcol)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _table_heatmap_png(df: pd.DataFrame, title: str, text_mode: str) -> bytes:
    if MATPLOTLIB_AVAILABLE:
        return _matplotlib_table_heatmap_png(df, title, text_mode)
    return _pillow_table_heatmap_png(df, title, text_mode)


@st.cache_data(show_spinner=False)
def _cached_std_tables(
    target_data: pd.DataFrame,
    hourly_data: pd.DataFrame,
    time_mode: str,
    start_h: int,
    end_h: int,
) -> Dict[str, pd.DataFrame]:
    return build_std_tables(
        target_data=target_data,
        hourly_data=hourly_data,
        time_mode=time_mode,
        start_h=start_h,
        end_h=end_h,
    )


@st.cache_data(show_spinner=False)
def _cached_month_daytime_table(target_data: pd.DataFrame, start_h: int, end_h: int) -> pd.DataFrame:
    return build_month_full_vs_daytime(target_data=target_data, day_start_hour=start_h, day_end_hour=end_h)


@st.cache_data(show_spinner=False)
def _cached_weekday_daytime_table(target_data: pd.DataFrame, start_h: int, end_h: int) -> pd.DataFrame:
    target = target_data.copy()
    base = pd.DataFrame({"weekday": list(range(7))})
    base["曜日"] = base["weekday"].map(lambda x: WEEKDAY_JP_ORDER[int(x)])

    weekday_stats = (
        target.groupby("weekday", as_index=False)["kw"]
        .agg(最高デマンド値_kW="max", 最低デマンド値_kW="min", 平均デマンド値_kW="mean")
    )
    full_energy = target.groupby("weekday", as_index=False)["kw"].sum().rename(columns={"kw": "全時間使用量_kWh"})
    full_energy["全時間使用量_kWh"] = full_energy["全時間使用量_kWh"] / 2.0

    day_mask = _time_range_mask_ui(target, start_h, end_h)
    daytime = target.loc[day_mask].copy()
    day_energy = daytime.groupby("weekday", as_index=False)["kw"].sum().rename(columns={"kw": "日中使用量_kWh"})
    day_energy["日中使用量_kWh"] = day_energy["日中使用量_kWh"] / 2.0

    out = (
        base.merge(weekday_stats, on="weekday", how="left")
        .merge(full_energy, on="weekday", how="left")
        .merge(day_energy, on="weekday", how="left")
        .sort_values("weekday")
    )
    out["日中利用率_%"] = np.where(
        out["全時間使用量_kWh"] > 0,
        out["日中使用量_kWh"] / out["全時間使用量_kWh"] * 100.0,
        np.nan,
    )
    return out


@st.cache_data(show_spinner=False)
def _cached_month_slot_matrix(target_data: pd.DataFrame, metric: str) -> pd.DataFrame:
    agg = {"MAX": "max", "MIN": "min", "AVE": "mean"}[metric]
    work = target_data.copy()
    slot_order = (
        work[["slot", "slot_label"]]
        .drop_duplicates()
        .sort_values("slot")
        ["slot_label"]
        .tolist()
    )
    pivot = (
        work.pivot_table(index="month", columns="slot_label", values="kw", aggfunc=agg)
        .reindex(index=list(range(1, 13)), columns=slot_order)
    )
    pivot.index = [f"{int(m)}月" for m in pivot.index]
    overall = work.groupby("slot_label")["kw"].agg(agg).reindex(slot_order).to_frame().T
    overall.index = ["全体"]
    return pd.concat([pivot, overall], axis=0)


def _time_range_mask_ui(df: pd.DataFrame, start_h: int, end_h: int) -> pd.Series:
    times = pd.to_datetime(df["datetime"]).dt.time
    start_t = pd.Timestamp(f"{int(start_h):02d}:00").time()
    end_t = pd.Timestamp(f"{int(end_h):02d}:00").time()
    if start_h <= end_h:
        return (times >= start_t) & (times < end_t)
    return (times >= start_t) | (times < end_t)


def _slice_columns_by_time(columns: list[str], start_col: str, end_col: str) -> list[str]:
    if start_col not in columns or end_col not in columns:
        return columns
    s_idx = columns.index(start_col)
    e_idx = columns.index(end_col)
    if s_idx <= e_idx:
        return columns[s_idx : e_idx + 1]
    return columns[s_idx:] + columns[: e_idx + 1]


def _pick_output_columns(columns: list[str], key_prefix: str) -> list[str]:
    if not columns:
        return columns

    has_half = any(":30" in c for c in columns)
    default_start = "06:00" if "06:00" in columns else columns[0]
    default_end = "18:00" if "18:00" in columns else columns[-1]
    if has_half and "18:30" in columns and default_end == "18:00":
        default_end = "18:00"

    mode = st.selectbox("PNG出力範囲", ["全列", "時刻指定"], index=1, key=f"{key_prefix}_png_range_mode")
    if mode == "全列":
        return columns

    c1, c2 = st.columns(2)
    with c1:
        start_col = st.selectbox("開始時刻", columns, index=columns.index(default_start), key=f"{key_prefix}_png_start")
    with c2:
        end_col = st.selectbox("終了時刻", columns, index=columns.index(default_end), key=f"{key_prefix}_png_end")
    picked = _slice_columns_by_time(columns, start_col, end_col)
    st.caption(f"出力対象列: {picked[0]} 〜 {picked[-1]}（{len(picked)}列）")
    return picked


def _calc_std_overall_series(
    target_data: pd.DataFrame,
    time_mode: str,
    start_h: int,
    end_h: int,
    cols_30: list[str],
    cols_60: list[str],
) -> tuple[pd.Series, pd.Series]:
    work = target_data.copy()
    if time_mode != "全時間":
        work = work.loc[_time_range_mask_ui(work, start_h, end_h)].copy()
    std30 = work.groupby("slot_label")["kw"].std(ddof=0).reindex(cols_30).fillna(0.0)
    std60_map: Dict[str, float] = {}
    for c in cols_60:
        hh = c.split(":")[0]
        c1 = f"{hh}:00"
        c2 = f"{hh}:30"
        if c1 in std30.index and c2 in std30.index:
            std60_map[c] = float((std30[c1] + std30[c2]) / 2.0)
        elif c1 in std30.index:
            std60_map[c] = float(std30[c1])
        elif c2 in std30.index:
            std60_map[c] = float(std30[c2])
        else:
            std60_map[c] = 0.0
    std60 = pd.Series(std60_map).reindex(cols_60).fillna(0.0)
    return std30, std60


def _fmt_corr(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "算出不可"
    return f"{float(value):.4f}"


def _three_color_rgb(value: float, vmin: float, vmed: float, vmax: float) -> tuple[int, int, int]:
    blue = np.array([44, 123, 182], dtype=float)
    white = np.array([255, 255, 255], dtype=float)
    red = np.array([215, 25, 28], dtype=float)

    if not np.isfinite(value):
        return (255, 255, 255)
    if vmax <= vmin:
        return (255, 255, 255)

    if value <= vmed:
        den = max(vmed - vmin, 1e-12)
        t = np.clip((value - vmin) / den, 0.0, 1.0)
        rgb = blue * (1.0 - t) + white * t
    else:
        den = max(vmax - vmed, 1e-12)
        t = np.clip((value - vmed) / den, 0.0, 1.0)
        rgb = white * (1.0 - t) + red * t

    r, g, b = rgb.astype(int)
    return (int(r), int(g), int(b))


def _text_color_for_mode(r: int, g: int, b: int, text_mode: str) -> str:
    if text_mode == "黒":
        return "#000000"
    if text_mode == "濃灰":
        return "#333333"
    # 自動コントラスト
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#000000" if luminance > 0.58 else "#FFFFFF"


def _cell_style(value: float, vmin: float, vmed: float, vmax: float, text_mode: str) -> str:
    r, g, b = _three_color_rgb(value, vmin, vmed, vmax)
    text = _text_color_for_mode(r, g, b, text_mode)
    return f"background-color: rgb({r},{g},{b}); color: {text}"


def _styled_heatmap(df: pd.DataFrame, text_mode: str = "自動コントラスト") -> pd.io.formats.style.Styler:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    vals = numeric.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return numeric.style.format("{:.2f}")

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    vmed = float(np.nanmedian(finite))
    return numeric.style.format("{:.2f}").applymap(lambda x: _cell_style(float(x), vmin, vmed, vmax, text_mode))


def _styled_heatmap_with_labels(df: pd.DataFrame, label_cols: list[str], text_mode: str = "自動コントラスト") -> pd.io.formats.style.Styler:
    work = df.copy()
    value_cols = [c for c in work.columns if c not in label_cols]
    numeric = work[value_cols].apply(pd.to_numeric, errors="coerce")
    vals = numeric.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    fmt = {c: "{:.2f}" for c in value_cols}
    styled = work.style.format(fmt)
    if finite.size == 0:
        return styled

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    vmed = float(np.nanmedian(finite))
    return styled.applymap(lambda x: _cell_style(float(x), vmin, vmed, vmax, text_mode), subset=value_cols)


def _line_chart_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_title: str = "kW") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines+markers",
            name=str(y_col),
            line=dict(color=_series_color(0), width=_line_width()),
            marker=dict(color=_series_color(0), size=6),
            hovertemplate=f"{x_col}: %{{x}}<br>{y_title}: %{{y:.2f}}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_title, margin=dict(l=40, r=20, t=50, b=50))
    return fig


def _render_line(df: pd.DataFrame, x_col: str, y_col: str, title: str, key_prefix: str, y_title: str = "kW") -> None:
    fig = _line_chart_figure(df, x_col, y_col, title, y_title)
    st.plotly_chart(fig, use_container_width=True)

    fallback = _fallback_line_png(
        df[x_col].astype(str).tolist(),
        pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).tolist(),
        title,
        y_label=y_title,
    )
    c1, c2 = st.columns(2)
    with c1:
        _download_png(fig, "PNGダウンロード", f"{key_prefix}.png", f"png_{key_prefix}", fallback)
    with c2:
        _download_csv(df, "CSVダウンロード", f"{key_prefix}.csv", f"csv_{key_prefix}")


def render_summary(summary: dict) -> None:
    st.subheader("サマリー")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("データ日数", f"{summary['data_days']} 日")
    c2.metric("最大デマンド", f"{summary['max_kw']:.2f} kW")
    c3.metric("平均デマンド", f"{summary['avg_kw']:.2f} kW")
    c4.metric("最大発生", pd.to_datetime(summary["max_datetime"]).strftime("%Y-%m-%d %H:%M"))


def render_daily_peak(daily_peak: pd.DataFrame) -> None:
    st.subheader("日別ピーク推移")
    view = daily_peak.copy()
    view["日付"] = pd.to_datetime(view["date"]).dt.strftime("%Y-%m-%d")
    view = view.rename(columns={"daily_peak_kw": "日別ピーク(kW)"})
    _render_line(view[["日付", "日別ピーク(kW)"]], "日付", "日別ピーク(kW)", "日別ピーク推移", "日別ピーク推移")


def render_avg_curve(avg_curve: pd.DataFrame) -> None:
    st.subheader("平均負荷曲線（48スロット）")
    view = avg_curve.rename(columns={"slot_label": "時刻", "avg_kw": "平均デマンド(kW)"})
    _render_line(view[["時刻", "平均デマンド(kW)"]], "時刻", "平均デマンド(kW)", "平均負荷曲線（48スロット）", "平均負荷曲線")


def render_weekday_curve(weekday_curve: pd.DataFrame) -> None:
    st.subheader("曜日別平均負荷曲線")
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    chosen = st.selectbox("曜日を選択", weekdays, key="weekday_select")
    one = weekday_curve[weekday_curve["weekday_jp"] == chosen].copy()
    if one.empty:
        st.info("選択した曜日のデータがありません。")
        return

    one = one.rename(columns={"slot_label": "時刻", "avg_kw": "平均デマンド(kW)"})
    _render_line(one[["時刻", "平均デマンド(kW)"]], "時刻", "平均デマンド(kW)", f"曜日別平均負荷曲線（{chosen}）", f"曜日別平均_{chosen}")


def render_histogram(histogram: pd.DataFrame) -> None:
    st.subheader("デマンド分布（ヒストグラム）")
    view = histogram.copy()
    view["レンジ"] = view.apply(lambda r: f"{r['bin_start']:.1f}〜{r['bin_end']:.1f} kW", axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=view["レンジ"],
            y=view["count"],
            marker_color="#4e79a7",
            hovertemplate="レンジ: %{x}<br>回数: %{y} 回<extra></extra>",
            name="回数",
        )
    )
    fig.update_layout(title="デマンド分布（ヒストグラム）", xaxis_title="レンジ", yaxis_title="回数", margin=dict(l=40, r=20, t=50, b=70))
    st.plotly_chart(fig, use_container_width=True)

    fallback = _fallback_bar_png(view["レンジ"].astype(str).tolist(), pd.to_numeric(view["count"], errors="coerce").fillna(0.0).tolist(), "デマンド分布（ヒストグラム）", "回数")
    c1, c2 = st.columns(2)
    with c1:
        _download_png(fig, "PNGダウンロード", "ヒストグラム.png", "png_hist", fallback)
    with c2:
        _download_csv(view[["レンジ", "count"]], "CSVダウンロード", "ヒストグラム.csv", "csv_hist")


def render_daytime(result: AnalysisResult) -> None:
    st.subheader("日中時間帯分析")
    st.caption(f"対象時間帯: {result.day_start_hour:02d}:00 - {result.day_end_hour:02d}:00")

    if result.daytime_summary is None:
        st.info("選択した時間帯のデータがありません。")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("時間帯平均", f"{result.daytime_summary['avg_kw']:.2f} kW")
    c2.metric("時間帯最大", f"{result.daytime_summary['max_kw']:.2f} kW")
    c3.metric("最大発生", pd.to_datetime(result.daytime_summary["max_datetime"]).strftime("%Y-%m-%d %H:%M"))

    view = result.daytime_curve.rename(columns={"slot_label": "時刻", "avg_kw": "平均デマンド(kW)"})
    _render_line(view[["時刻", "平均デマンド(kW)"]], "時刻", "平均デマンド(kW)", "日中時間帯 平均負荷曲線", "日中時間帯平均")


def render_annual_monthly_average(result: AnalysisResult) -> None:
    st.subheader("グラフ（平均値）")
    months = sorted(result.monthly_avg_curve["month"].dropna().unique().tolist())
    if not months:
        st.info("月別データがありません。")
        return

    selected_month = st.selectbox("月別デマンド平均の対象月", months, format_func=lambda x: f"{int(x)}月", key="avg_month")

    col1, col2 = st.columns(2)
    with col1:
        annual = result.annual_avg_curve.rename(columns={"slot_label": "時刻", "avg_kw": "平均デマンド(kW)"})
        _render_line(annual[["時刻", "平均デマンド(kW)"]], "時刻", "平均デマンド(kW)", "年間デマンド平均", "年間デマンド平均")

    with col2:
        one = result.monthly_avg_curve[result.monthly_avg_curve["month"] == selected_month].copy()
        one = one.rename(columns={"slot_label": "時刻", "avg_kw": "平均デマンド(kW)"})
        _render_line(one[["時刻", "平均デマンド(kW)"]], "時刻", "平均デマンド(kW)", f"月別デマンド平均（{int(selected_month)}月）", f"月別平均_{int(selected_month)}月")

    st.markdown("**1～12月デマンド平均**")
    fig = go.Figure()
    lines_for_png: Dict[str, list[float]] = {}
    month_curve = result.monthly_avg_curve.sort_values(["month", "slot"])
    for idx, (month, grp) in enumerate(month_curve.groupby("month")):
        label = f"{int(month)}月"
        y_vals = pd.to_numeric(grp["avg_kw"], errors="coerce").fillna(0.0).tolist()
        lines_for_png[label] = y_vals
        fig.add_trace(
            go.Scatter(
                x=grp["slot_label"],
                y=grp["avg_kw"],
                mode="lines",
                name=label,
                line=dict(color=_series_color(idx), width=_line_width()),
                hovertemplate="区分: %{fullData.name}<br>時刻: %{x}<br>平均デマンド: %{y:.2f} kW<extra></extra>",
            )
        )

    annual = result.annual_avg_curve.sort_values("slot")
    fig.add_trace(
        go.Scatter(
            x=annual["slot_label"],
            y=annual["avg_kw"],
            mode="lines",
            name="全体",
            line=dict(color="#111111", width=max(_line_width() + 0.5, 3.0), dash="dot"),
            hovertemplate="区分: 全体<br>時刻: %{x}<br>平均デマンド: %{y:.2f} kW<extra></extra>",
        )
    )
    lines_for_png["全体"] = pd.to_numeric(annual["avg_kw"], errors="coerce").fillna(0.0).tolist()
    fig.update_layout(
        title="1～12月デマンド平均",
        xaxis_title="時刻",
        yaxis_title="平均デマンド(kW)",
        margin=dict(l=40, r=20, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    csv_wide = (
        month_curve.pivot_table(index="slot_label", columns="month", values="avg_kw", aggfunc="mean")
        .rename(columns=lambda x: f"{int(x)}月")
        .reindex(index=annual["slot_label"].tolist())
    )
    csv_wide["全体"] = annual.set_index("slot_label")["avg_kw"].reindex(csv_wide.index)
    csv_out = csv_wide.reset_index().rename(columns={"slot_label": "時刻"})
    fallback = _fallback_multi_line_png(lines_for_png, "1～12月デマンド平均", "kW")
    d1, d2 = st.columns(2)
    with d1:
        _download_png(fig, "PNGダウンロード", "1～12月デマンド平均.png", "png_monthly_overlay", fallback)
    with d2:
        _download_csv(csv_out, "CSVダウンロード", "1～12月デマンド平均.csv", "csv_monthly_overlay")


def render_monthly_actual(result: AnalysisResult) -> None:
    st.subheader("グラフ（月別実績）")
    months = sorted(result.monthly_daily_curves["month"].dropna().unique().tolist())
    if not months:
        st.info("月別実績データがありません。")
        return

    selected_month = st.selectbox("対象月", months, format_func=lambda x: f"{int(x)}月", key="actual_month")
    one_month = result.monthly_daily_curves[result.monthly_daily_curves["month"] == selected_month].copy()
    if one_month.empty:
        st.info("該当月のデータがありません。")
        return

    day_options = sorted(pd.to_datetime(one_month["date"]).dt.date.unique().tolist())
    chosen_days = st.multiselect("表示日（未選択で全日）", day_options, default=[], format_func=lambda d: d.strftime("%m/%d"), key="actual_days")
    if chosen_days:
        one_month = one_month[pd.to_datetime(one_month["date"]).dt.date.isin(chosen_days)].copy()

    one_month["日付"] = pd.to_datetime(one_month["date"]).dt.strftime("%Y-%m-%d")
    one_month = one_month.sort_values(["日付", "slot"])  # type: ignore[arg-type]

    fig = go.Figure()
    lines_for_png: Dict[str, list[float]] = {}
    for idx, (d, grp) in enumerate(one_month.groupby("日付")):
        grp = grp.sort_values("slot")
        y_vals = pd.to_numeric(grp["kw"], errors="coerce").fillna(0.0).tolist()
        lines_for_png[str(d)] = y_vals
        fig.add_trace(
            go.Scatter(
                x=grp["slot_label"],
                y=grp["kw"],
                mode="lines",
                name=str(d),
                line=dict(color=_series_color(idx), width=_line_width()),
                hovertemplate=f"日付: {d}<br>時刻: %{{x}}<br>デマンド: %{{y:.2f}} kW<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{int(selected_month)}月デマンド値",
        xaxis_title="時刻",
        yaxis_title="デマンド(kW)",
        showlegend=len(lines_for_png) <= 15,
        margin=dict(l=40, r=20, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    csv_out = one_month[["日付", "slot_label", "kw"]].rename(columns={"slot_label": "時刻", "kw": "デマンド(kW)"})
    fallback = _fallback_multi_line_png(lines_for_png, f"{int(selected_month)}月デマンド値", "kW")
    c1, c2 = st.columns(2)
    with c1:
        _download_png(fig, "PNGダウンロード", f"月別実績_{int(selected_month)}月.png", "png_month_actual", fallback)
    with c2:
        _download_csv(csv_out, "CSVダウンロード", f"月別実績_{int(selected_month)}月.csv", "csv_month_actual")


def _render_std_table(title: str, table: pd.DataFrame, key_prefix: str, text_mode: str) -> None:
    st.markdown(f"**{title}**")
    if table.empty:
        st.info("データがありません。")
        return

    show = table.copy()
    show.index.name = "区分"
    st.dataframe(_styled_heatmap(show, text_mode=text_mode), use_container_width=True, height=380)

    selected_cols = _pick_output_columns(show.columns.tolist(), key_prefix=key_prefix)
    png_target = show.reindex(columns=selected_cols)
    png = _table_heatmap_png(png_target, title=title, text_mode=text_mode)

    csv_same_range = st.checkbox("CSVも同じ時間範囲で出力", value=False, key=f"{key_prefix}_csv_same_range")
    csv_target = png_target if csv_same_range else show
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("PNGダウンロード", data=png, file_name=f"{key_prefix}.png", mime="image/png", key=f"png_{key_prefix}")
    with c2:
        _download_csv(csv_target.reset_index(), "CSVダウンロード", f"{key_prefix}.csv", f"csv_{key_prefix}")


def render_std_analysis(result: AnalysisResult) -> None:
    st.subheader("月別・曜日別 標準偏差")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**標準偏差の定義**")
        st.caption("STDEV.P（母標準偏差 / ddof=0）固定")
    with c2:
        time_mode = st.selectbox("時間帯モード", ["全時間", "任意時間帯"], index=0, key="std_time_mode")
    with c3:
        text_mode = st.selectbox("数値文字色", ["自動コントラスト", "黒", "濃灰"], index=0, key="std_text_mode")

    start_h, end_h = 9, 18
    if time_mode == "任意時間帯":
        start_h, end_h = st.slider("任意時間帯（時）", min_value=0, max_value=24, value=(9, 18), step=1, key="std_time_slider")
        st.caption(f"対象時間帯: {start_h:02d}:00 - {end_h:02d}:00")

    std_tables = _cached_std_tables(
        target_data=result.target_data,
        hourly_data=result.hourly_data,
        time_mode=time_mode,
        start_h=start_h,
        end_h=end_h,
    )

    tabs = st.tabs(["月別30分", "月別1時間", "曜日別30分", "曜日別1時間"])
    with tabs[0]:
        _render_std_table(
            f"月別標準偏差（30分 / {time_mode}）",
            std_tables["month_30"],
            f"月別標準偏差_30分_{time_mode}_STDEVP",
            text_mode=text_mode,
        )
    with tabs[1]:
        _render_std_table(
            f"月別標準偏差（1時間 / {time_mode}）",
            std_tables["month_60"],
            f"月別標準偏差_1時間_{time_mode}_STDEVP",
            text_mode=text_mode,
        )
    with tabs[2]:
        _render_std_table(
            f"曜日別標準偏差（30分 / {time_mode}）",
            std_tables["weekday_30"],
            f"曜日別標準偏差_30分_{time_mode}_STDEVP",
            text_mode=text_mode,
        )
    with tabs[3]:
        _render_std_table(
            f"曜日別標準偏差（1時間 / {time_mode}）",
            std_tables["weekday_60"],
            f"曜日別標準偏差_1時間_{time_mode}_STDEVP",
            text_mode=text_mode,
        )

    std30_overall, std60_overall = _calc_std_overall_series(
        target_data=result.target_data,
        time_mode=time_mode,
        start_h=start_h,
        end_h=end_h,
        cols_30=std_tables["month_30"].columns.tolist(),
        cols_60=std_tables["month_60"].columns.tolist(),
    )

    st.markdown("### 標準偏差 折れ線グラフ")
    g1, g2 = st.columns(2)
    with g1:
        graph_group = st.selectbox("区分", ["月別", "曜日別"], index=0, key="std_graph_group")
    with g2:
        graph_interval = st.radio("粒度", ["30分", "1時間"], horizontal=True, key="std_graph_interval")

    if graph_group == "月別" and graph_interval == "30分":
        base = std_tables["month_30"]
        overall = std30_overall
    elif graph_group == "月別" and graph_interval == "1時間":
        base = std_tables["month_60"]
        overall = std60_overall
    elif graph_group == "曜日別" and graph_interval == "30分":
        base = std_tables["weekday_30"]
        overall = std30_overall
    else:
        base = std_tables["weekday_60"]
        overall = std60_overall

    fig = go.Figure()
    lines_for_png: Dict[str, list[float]] = {}
    for idx, (label, row) in enumerate(base.iterrows()):
        y_vals = pd.to_numeric(row, errors="coerce").fillna(0.0).tolist()
        lines_for_png[str(label)] = y_vals
        fig.add_trace(
            go.Scatter(
                x=base.columns.tolist(),
                y=y_vals,
                mode="lines+markers",
                name=str(label),
                line=dict(color=_series_color(idx), width=_line_width()),
                marker=dict(size=6),
                hovertemplate="区分: %{fullData.name}<br>時刻: %{x}<br>標準偏差: %{y:.2f} kW<extra></extra>",
            )
        )

    overall_vals = pd.to_numeric(overall.reindex(base.columns.tolist()), errors="coerce").fillna(0.0).tolist()
    lines_for_png["全体"] = overall_vals
    fig.add_trace(
        go.Scatter(
            x=base.columns.tolist(),
            y=overall_vals,
            mode="lines+markers",
            name="全体",
            line=dict(color="#111111", width=max(_line_width() + 0.5, 3.0), dash="dot"),
            marker=dict(size=6, color="#111111"),
            hovertemplate="区分: 全体<br>時刻: %{x}<br>標準偏差: %{y:.2f} kW<extra></extra>",
        )
    )

    std_title = "月別　時間毎デマンド値　標準偏差" if graph_group == "月別" else "曜日別　時間毎デマンド値　標準偏差"
    fig.update_layout(
        title=f"{std_title}（{graph_interval} / {time_mode}）",
        xaxis_title="時刻",
        yaxis_title="標準偏差(kW)",
        margin=dict(l=40, r=20, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    csv_out = base.T.copy()
    csv_out["全体"] = pd.Series(overall_vals, index=base.columns.tolist())
    csv_out = csv_out.reset_index().rename(columns={"index": "時刻"})
    fallback = _fallback_multi_line_png(lines_for_png, f"{std_title}（{graph_interval}）", "kW")
    d1, d2 = st.columns(2)
    with d1:
        _download_png(fig, "PNGダウンロード", f"標準偏差折れ線_{graph_group}_{graph_interval}.png", "png_std_line", fallback)
    with d2:
        _download_csv(csv_out, "CSVダウンロード", f"標準偏差折れ線_{graph_group}_{graph_interval}.csv", "csv_std_line")


def _render_daytime_demand_block(table: pd.DataFrame, category_col: str, title: str, key_prefix: str, start_h: int, end_h: int) -> None:
    show_cols = [
        category_col,
        "最高デマンド値(kW)",
        "最低デマンド値(kW)",
        "平均デマンド値(kW)",
        "全時間使用量(kWh)",
        "日中使用量(kWh)",
        "日中利用率(%)",
    ]
    table = table[show_cols].copy()
    st.dataframe(_styled_heatmap_with_labels(table, label_cols=[category_col]), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=table[category_col],
            y=table["全時間使用量(kWh)"],
            name="全時間使用量",
            hovertemplate=f"{category_col}: %{{x}}<br>全時間使用量: %{{y:.2f}} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=table[category_col],
            y=table["日中使用量(kWh)"],
            name="日中使用量",
            hovertemplate=f"{category_col}: %{{x}}<br>日中使用量: %{{y:.2f}} kWh<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"全時間使用量 vs 日中使用量（{start_h:02d}:00-{end_h:02d}:00）",
        barmode="group",
        xaxis_title=category_col,
        yaxis_title="使用量(kWh)",
        margin=dict(l=40, r=20, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    corr_base = table[["最高デマンド値(kW)", "全時間使用量(kWh)", "日中使用量(kWh)"]].dropna()
    corr1 = float(corr_base["最高デマンド値(kW)"].corr(corr_base["全時間使用量(kWh)"])) if len(corr_base) >= 2 else np.nan
    corr2 = float(corr_base["最高デマンド値(kW)"].corr(corr_base["日中使用量(kWh)"])) if len(corr_base) >= 2 else np.nan
    m1, m2 = st.columns(2)
    m1.metric("相関① 最高デマンド値 × 全時間使用量", _fmt_corr(corr1))
    m2.metric("相関② 最高デマンド値 × 日中使用量", _fmt_corr(corr2))

    fallback = _fallback_bar_png(
        table[category_col].astype(str).tolist(),
        pd.to_numeric(table["全時間使用量(kWh)"], errors="coerce").fillna(0.0).tolist(),
        title,
        "kWh",
    )
    c1, c2 = st.columns(2)
    with c1:
        _download_png(fig, "PNGダウンロード", f"{title}.png", f"png_{key_prefix}", fallback)
    with c2:
        _download_csv(table, "CSVダウンロード", f"{title}.csv", f"csv_{key_prefix}")


def render_full_vs_daytime(result: AnalysisResult) -> None:
    st.subheader("月別日中デマンド")
    start_h, end_h = st.slider("日中時間帯（時）", min_value=0, max_value=24, value=(6, 18), step=1, key="month_daytime_slider")
    table_raw = _cached_month_daytime_table(result.target_data, start_h, end_h)
    if table_raw.empty:
        st.info("データがありません。")
        return

    table = table_raw.rename(
        columns={
            "月": "月",
            "最高デマンド値_kW": "最高デマンド値(kW)",
            "最低デマンド値_kW": "最低デマンド値(kW)",
            "平均デマンド値_kW": "平均デマンド値(kW)",
            "全時間使用量_kWh": "全時間使用量(kWh)",
            "日中使用量_kWh": "日中使用量(kWh)",
            "日中利用率_%": "日中利用率(%)",
        }
    )
    _render_daytime_demand_block(table, category_col="月", title="月別日中デマンド", key_prefix="month_daytime", start_h=start_h, end_h=end_h)


def render_weekday_daytime(result: AnalysisResult) -> None:
    st.subheader("曜日別日中デマンド")
    start_h, end_h = st.slider("日中時間帯（時）", min_value=0, max_value=24, value=(6, 18), step=1, key="weekday_daytime_slider")
    table_raw = _cached_weekday_daytime_table(result.target_data, start_h, end_h)
    if table_raw.empty:
        st.info("データがありません。")
        return

    table = table_raw.rename(
        columns={
            "最高デマンド値_kW": "最高デマンド値(kW)",
            "最低デマンド値_kW": "最低デマンド値(kW)",
            "平均デマンド値_kW": "平均デマンド値(kW)",
            "全時間使用量_kWh": "全時間使用量(kWh)",
            "日中使用量_kWh": "日中使用量(kWh)",
            "日中利用率_%": "日中利用率(%)",
        }
    )
    _render_daytime_demand_block(table, category_col="曜日", title="曜日別日中デマンド", key_prefix="weekday_daytime", start_h=start_h, end_h=end_h)


def render_slot_max_and_count(result: AnalysisResult) -> None:
    st.subheader("時間別デマンド回数（時間別最高デマンド値）")
    data = result.slot_max_and_count.copy().sort_values("slot")
    if data.empty:
        st.info("データがありません。")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data["slot_label"],
            y=data["max_count"],
            name="最高値記録回数",
            yaxis="y2",
            marker_color="rgba(89, 161, 79, 0.28)",
            width=0.55,
            hovertemplate="時刻: %{x}<br>最高値記録回数: %{y} 回<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["slot_label"],
            y=data["max_kw"],
            mode="lines+markers",
            name="最高デマンド値",
            line=dict(color=_series_color(0), width=max(_line_width() + 1.0, 3.5)),
            marker=dict(size=7, color=_series_color(0)),
            hovertemplate="時刻: %{x}<br>最高デマンド値: %{y:.2f} kW<extra></extra>",
        )
    )
    fig.update_layout(
        title="時間別デマンド回数（時間別最高デマンド値）",
        xaxis_title="時刻",
        yaxis=dict(title="最高デマンド値(kW)"),
        yaxis2=dict(title="回数", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    fallback = _fallback_dual_png(
        data["slot_label"].astype(str).tolist(),
        pd.to_numeric(data["max_kw"], errors="coerce").fillna(0.0).tolist(),
        pd.to_numeric(data["max_count"], errors="coerce").fillna(0.0).tolist(),
        "時間別デマンド回数（時間別最高デマンド値）",
    )
    out = data.rename(columns={"slot_label": "時刻", "max_kw": "最高デマンド値(kW)", "max_count": "最高値記録回数"})
    c1, c2 = st.columns(2)
    with c1:
        _download_png(fig, "PNGダウンロード", "時間別デマンド回数（時間別最高デマンド値）.png", "png_slot_count", fallback)
    with c2:
        _download_csv(out[["時刻", "最高デマンド値(kW)", "最高値記録回数"]], "CSVダウンロード", "時間別デマンド回数（時間別最高デマンド値）.csv", "csv_slot_count")


def render_demand_range(result: AnalysisResult) -> None:
    st.subheader("デマンドレンジ")
    st.markdown("**デマンドレンジ別回数（既存）**")
    width = st.number_input("レンジ幅 (kW)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0, key="range_width")
    base_table = summarize_demand_ranges(result.target_data["kw"], bin_width_kw=float(width))
    if not base_table.empty:
        st.dataframe(base_table, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=base_table["レンジ"],
                y=base_table["回数"],
                marker_color="#4e79a7",
                hovertemplate="レンジ: %{x}<br>回数: %{y} 回<extra></extra>",
                name="回数",
            )
        )
        fig.update_layout(title="デマンドレンジ別回数", xaxis_title="レンジ", yaxis_title="回数", margin=dict(l=40, r=20, t=50, b=70))
        st.plotly_chart(fig, use_container_width=True)

        fallback = _fallback_bar_png(
            base_table["レンジ"].astype(str).tolist(),
            pd.to_numeric(base_table["回数"], errors="coerce").fillna(0.0).tolist(),
            "デマンドレンジ別回数",
            "回数",
        )
        b1, b2 = st.columns(2)
        with b1:
            _download_png(fig, "PNGダウンロード", "デマンドレンジ別回数.png", "png_range_base", fallback)
        with b2:
            _download_csv(base_table, "CSVダウンロード", "デマンドレンジ別回数.csv", "csv_range_base")

    st.markdown("**月別×時刻 虹色テーブル（MAX / MIN / AVE）**")
    text_mode = st.selectbox("数値文字色", ["自動コントラスト", "黒", "濃灰"], index=0, key="range_text_mode")
    tabs = st.tabs(["MAX", "MIN", "AVE"])
    for metric, tab in zip(["MAX", "MIN", "AVE"], tabs):
        with tab:
            table = _cached_month_slot_matrix(result.target_data, metric=metric)
            if table.empty:
                st.info("データがありません。")
                continue

            show = table.copy()
            show.index.name = "区分"
            st.dataframe(_styled_heatmap(show, text_mode=text_mode), use_container_width=True, height=420)
            selected_cols = _pick_output_columns(show.columns.tolist(), key_prefix=f"range_{metric}")
            png_target = show.reindex(columns=selected_cols)
            png = _table_heatmap_png(png_target, title=f"デマンドレンジ {metric}（月×時刻）", text_mode=text_mode)
            csv_same_range = st.checkbox("CSVも同じ時間範囲で出力", value=False, key=f"range_{metric}_csv_same_range")
            csv_target = png_target if csv_same_range else show
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("PNGダウンロード", data=png, file_name=f"デマンドレンジ_{metric}.png", mime="image/png", key=f"png_range_{metric}")
            with c2:
                _download_csv(csv_target.reset_index(), "CSVダウンロード", f"デマンドレンジ_{metric}.csv", f"csv_range_{metric}")


def render_peak_cut(result: AnalysisResult) -> None:
    st.subheader("ピークカット目安試算（参考・簡易試算）")
    current_peak = float(result.summary["max_kw"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_peak = st.number_input("目標ピーク(kW)", min_value=0.0, value=max(current_peak * 0.9, 0.0), step=1.0, key="pc_target")
    with c2:
        unit_price = st.number_input("基本料金単価(円/kW・月)", min_value=0.0, value=1800.0, step=100.0, key="pc_unit")
    with c3:
        months = st.number_input("適用月数", min_value=1, max_value=12, value=12, step=1, key="pc_months")
    with c4:
        initial_cost = st.number_input("初期費用(円)", min_value=0.0, value=0.0, step=10000.0, key="pc_initial")

    calc = estimate_peak_cut(current_peak, target_peak, unit_price, int(months), initial_cost)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("削減量", f"{calc['削減量(kW)']:.2f} kW")
    m2.metric("月額削減効果", f"{calc['月額削減効果(円)']:.0f} 円")
    m3.metric("年間削減効果", f"{calc['年間削減効果(円)']:.0f} 円")
    payback = calc["投資回収年(年)"]
    m4.metric("投資回収年", "算出不可" if payback is None else f"{payback:.2f} 年")

    _download_csv(pd.DataFrame([calc]), "試算CSVダウンロード", "ピークカット試算.csv", "csv_peak_cut")


def _render_filtered_table(df: pd.DataFrame, key_prefix: str) -> None:
    if df.empty:
        st.info("該当データがありません。")
        return
    st.dataframe(df, use_container_width=True, height=300)
    _download_csv(df, "CSVダウンロード", f"{key_prefix}.csv", f"csv_{key_prefix}")


def render_sheet_like_views(result: AnalysisResult) -> None:
    st.subheader("月毎データ")
    data = result.target_data.copy()
    data["date"] = pd.to_datetime(data["date"])

    tab_m, tab_w = st.tabs(["月毎データ", "曜日別データ"])

    with tab_m:
        month = st.selectbox("表示月", sorted(data["month"].unique()), format_func=lambda x: f"{int(x)}月", key="sheet_month")
        month_df = data[data["month"] == month].copy()
        daily = (
            month_df.groupby("date", as_index=False)["kw"]
            .agg(日最大デマンド_kW="max", 日平均デマンド_kW="mean")
            .sort_values("date")
        )
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        _render_filtered_table(daily, f"月毎データ_{int(month)}月_日別")

        if not month_df.empty:
            date_options = sorted(month_df["date"].dt.date.unique().tolist())
            selected_date = st.selectbox("30分値を表示する日", date_options, key="sheet_month_date")
            slots = month_df[month_df["date"].dt.date == selected_date][["slot_label", "kw"]].rename(columns={"slot_label": "時刻", "kw": "デマンド(kW)"})
            _render_filtered_table(slots, f"月毎データ_{int(month)}月_{selected_date}")

    with tab_w:
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        weekday = st.selectbox("曜日", weekdays, key="sheet_weekday")
        month_filter = st.selectbox("月フィルタ", ["全月"] + [f"{m}月" for m in sorted(data["month"].unique())], key="sheet_weekday_month")

        wdf = data[data["weekday_jp"] == weekday].copy()
        if month_filter != "全月":
            mf = int(month_filter.replace("月", ""))
            wdf = wdf[wdf["month"] == mf]

        wdaily = (
            wdf.groupby("date", as_index=False)["kw"]
            .agg(日最大デマンド_kW="max", 日平均デマンド_kW="mean")
            .sort_values("date")
        )
        wdaily["date"] = pd.to_datetime(wdaily["date"]).dt.strftime("%Y-%m-%d")
        _render_filtered_table(wdaily, f"曜日別データ_{weekday}")

        if not wdf.empty:
            date_options = sorted(pd.to_datetime(wdf["date"]).dt.date.unique().tolist())
            selected_date = st.selectbox("30分値を表示する日", date_options, key="sheet_weekday_date")
            slots = wdf[pd.to_datetime(wdf["date"]).dt.date == selected_date][["slot_label", "kw"]].rename(columns={"slot_label": "時刻", "kw": "デマンド(kW)"})
            _render_filtered_table(slots, f"曜日別データ_{weekday}_{selected_date}")


def render_input_data_heatmap(result: AnalysisResult) -> None:
    st.subheader("入力データ可視化")
    data = result.target_data.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.date

    min_d = min(data["date"])
    max_d = max(data["date"])

    c1, c2, c3 = st.columns(3)
    with c1:
        date_from = st.date_input("表示開始日", value=min_d, min_value=min_d, max_value=max_d, key="inp_heat_from")
    with c2:
        date_to = st.date_input("表示終了日", value=max_d, min_value=min_d, max_value=max_d, key="inp_heat_to")
    with c3:
        max_rows = int(st.number_input("表示行数上限", min_value=10, max_value=400, value=90, step=10, key="inp_heat_rows"))

    months = sorted(data["month"].dropna().unique().tolist())
    selected_months = st.multiselect("表示月（未選択=全月）", months, default=[], format_func=lambda x: f"{int(x)}月", key="inp_heat_months")
    text_mode = st.selectbox("数値文字色", ["自動コントラスト", "黒", "濃灰"], index=0, key="inp_heat_text")

    if date_from > date_to:
        st.warning("表示開始日は表示終了日以前にしてください。")
        return

    view = data[(data["date"] >= date_from) & (data["date"] <= date_to)].copy()
    if selected_months:
        view = view[view["month"].isin(selected_months)].copy()
    if view.empty:
        st.info("条件に一致するデータがありません。")
        return

    slot_order = view[["slot", "slot_label"]].drop_duplicates().sort_values("slot")["slot_label"].tolist()
    pivot = (
        view.pivot_table(index="date", columns="slot_label", values="kw", aggfunc="mean")
        .reindex(columns=slot_order)
        .sort_index()
    )
    if len(pivot) > max_rows:
        pivot = pivot.tail(max_rows)

    pivot_show = pivot.copy()
    pivot_show.index = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in pivot_show.index]
    st.dataframe(_styled_heatmap(pivot_show, text_mode=text_mode), use_container_width=True, height=430)

    title = f"入力データ色付き表 ({pivot_show.index.min()}〜{pivot_show.index.max()})"
    png = _table_heatmap_png(pivot_show, title=title, text_mode=text_mode)
    c_png, c_csv = st.columns(2)
    with c_png:
        st.download_button("PNGダウンロード", data=png, file_name="入力データ色付き表.png", mime="image/png", key="png_input_heat")
    with c_csv:
        _download_csv(pivot_show.reset_index().rename(columns={"index": "日付"}), "CSVダウンロード", "入力データ色付き表.csv", "csv_input_heat")


def render_selected_sections(
    result: AnalysisResult,
    show_summary: bool,
    show_weekday_curve: bool,
    show_avg_ext: bool,
    show_monthly_actual: bool,
    show_weekday_daytime_graph: bool,
    show_std: bool,
    show_month_daytime_graph: bool,
    show_slot_max_count: bool,
    show_range: bool,
    show_peak_cut: bool,
    show_sheet_like: bool,
    show_input_heatmap: bool,
) -> None:
    if show_summary:
        render_summary(result.summary)
    if show_weekday_curve:
        render_weekday_curve(result.weekday_curve)

    if show_avg_ext:
        render_annual_monthly_average(result)
    if show_monthly_actual:
        render_monthly_actual(result)
    if show_weekday_daytime_graph:
        render_weekday_daytime(result)
    if show_std:
        render_std_analysis(result)
    if show_month_daytime_graph:
        render_full_vs_daytime(result)
    if show_slot_max_count:
        render_slot_max_and_count(result)
    if show_range:
        render_demand_range(result)
    if show_peak_cut:
        render_peak_cut(result)
    if show_sheet_like:
        render_sheet_like_views(result)
    if show_input_heatmap:
        render_input_data_heatmap(result)
