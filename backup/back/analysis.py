from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

WEEKDAY_JP = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
WEEKDAY_ORDER = [0, 1, 2, 3, 4, 5, 6]
SLOT_LABELS_30 = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
SLOT_LABELS_60 = [f"{h:02d}:00" for h in range(24)]
MONTH_ORDER = list(range(1, 13))


@dataclass(frozen=True)
class AnalysisResult:
    summary: Dict[str, object]
    daily_peak: pd.DataFrame
    avg_curve: pd.DataFrame
    weekday_curve: pd.DataFrame
    histogram: pd.DataFrame
    target_data: pd.DataFrame
    hourly_data: pd.DataFrame
    annual_avg_curve: pd.DataFrame
    monthly_avg_curve: pd.DataFrame
    monthly_daily_curves: pd.DataFrame
    month_std_30_full: pd.DataFrame
    month_std_30_daytime: pd.DataFrame
    month_std_60_full: pd.DataFrame
    month_std_60_daytime: pd.DataFrame
    weekday_std_30_full: pd.DataFrame
    weekday_std_30_daytime: pd.DataFrame
    weekday_std_60_full: pd.DataFrame
    weekday_std_60_daytime: pd.DataFrame
    month_full_vs_daytime: pd.DataFrame
    slot_max_and_count: pd.DataFrame
    demand_range_default: pd.DataFrame
    daytime_enabled: bool
    day_start_hour: int
    day_end_hour: int
    daytime_summary: Optional[Dict[str, object]]
    daytime_curve: pd.DataFrame


def _filter_period(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df.loc[mask].copy()


def _time_range_mask(df: pd.DataFrame, start_h: int, end_h: int) -> pd.Series:
    start_t = time(start_h, 0)
    end_t = time(end_h, 0)
    times = df["datetime"].dt.time
    if start_h <= end_h:
        return (times >= start_t) & (times < end_t)
    return (times >= start_t) | (times < end_t)


def _build_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["hour_slot"] = (work["slot"] // 2).astype(int)
    work["hour_label"] = work["hour_slot"].map(lambda x: SLOT_LABELS_60[x])
    return (
        work.groupby(["date", "hour_slot", "hour_label"], as_index=False)["kw"]
        .mean()
        .sort_values(["date", "hour_slot"])
    )


def _make_std_pivot(
    df: pd.DataFrame,
    row_col: str,
    row_order: List[int],
    col_col: str,
    col_order: List[str],
    row_labeler,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=[row_labeler(v) for v in row_order], columns=col_order).fillna(0.0)

    pivot = (
        df.groupby([row_col, col_col])["kw"]
        .std(ddof=0)
        .unstack(col_col)
        .reindex(index=row_order, columns=col_order)
        .fillna(0.0)
    )
    pivot.index = [row_labeler(v) for v in pivot.index]
    return pivot


def _daytime_halfhour_labels(start_h: int, end_h: int) -> List[str]:
    labels: List[str] = []
    for h in range(24):
        in_range = (start_h <= h < end_h) if start_h <= end_h else (h >= start_h or h < end_h)
        if in_range:
            labels.extend([f"{h:02d}:00", f"{h:02d}:30"])
    return labels


def _daytime_hour_labels(start_h: int, end_h: int) -> List[str]:
    labels: List[str] = []
    for h in range(24):
        in_range = (start_h <= h < end_h) if start_h <= end_h else (h >= start_h or h < end_h)
        if in_range:
            labels.append(f"{h:02d}:00")
    return labels


def _std30_to_std60(std30: pd.DataFrame, hour_cols: List[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=std30.index)
    for h in range(24):
        c1 = f"{h:02d}:00"
        c2 = f"{h:02d}:30"
        if c1 in std30.columns and c2 in std30.columns:
            result[c1] = (std30[c1] + std30[c2]) / 2.0
        elif c1 in std30.columns:
            result[c1] = std30[c1]
        elif c2 in std30.columns:
            result[c1] = std30[c2]
    return result.reindex(columns=hour_cols).fillna(0.0)


def _time_filter_mask_by_mode(df: pd.DataFrame, time_mode: str, start_h: int, end_h: int) -> pd.Series:
    if time_mode == "全時間":
        return pd.Series(True, index=df.index)
    return _time_range_mask(df, start_h, end_h)


def build_std_tables(
    target_data: pd.DataFrame,
    hourly_data: pd.DataFrame,
    time_mode: str = "全時間",
    start_h: int = 9,
    end_h: int = 18,
) -> Dict[str, pd.DataFrame]:
    target = target_data.copy()
    _ = hourly_data  # 互換性維持のため引数は残す

    tmask = _time_filter_mask_by_mode(target, time_mode, start_h, end_h)
    t = target.loc[tmask].copy()

    if time_mode == "全時間":
        cols_30 = SLOT_LABELS_30
        cols_60 = SLOT_LABELS_60
    else:
        cols_30 = _daytime_halfhour_labels(start_h, end_h)
        cols_60 = _daytime_hour_labels(start_h, end_h)

    month_std_30 = _make_std_pivot(t, "month", MONTH_ORDER, "slot_label", cols_30, lambda v: f"{v}月")
    weekday_std_30 = _make_std_pivot(t, "weekday", WEEKDAY_ORDER, "slot_label", cols_30, lambda v: WEEKDAY_JP[v])
    month_std_60 = _std30_to_std60(month_std_30, cols_60)
    weekday_std_60 = _std30_to_std60(weekday_std_30, cols_60)

    return {
        "month_30": month_std_30,
        "month_60": month_std_60,
        "weekday_30": weekday_std_30,
        "weekday_60": weekday_std_60,
    }


def summarize_demand_ranges(values: pd.Series, bin_width_kw: float = 50.0) -> pd.DataFrame:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return pd.DataFrame(columns=["レンジ", "回数", "割合(%)"])

    width = max(float(bin_width_kw), 1.0)
    vmin = max(0.0, float(np.floor(numeric.min() / width) * width))
    vmax = float(np.ceil(numeric.max() / width) * width)
    if vmax <= vmin:
        vmax = vmin + width

    bins = np.arange(vmin, vmax + width, width)
    if len(bins) < 2:
        bins = np.array([vmin, vmin + width])

    cats = pd.cut(numeric, bins=bins, right=False, include_lowest=True)
    counts = cats.value_counts(sort=False)
    total = counts.sum()

    rows = []
    for interval, cnt in counts.items():
        left = float(interval.left)
        right = float(interval.right)
        label = f"{left:.0f}〜{right:.0f} kW"
        ratio = (float(cnt) / float(total) * 100.0) if total > 0 else 0.0
        rows.append({"レンジ": label, "回数": int(cnt), "割合(%)": ratio})

    return pd.DataFrame(rows)


def build_month_full_vs_daytime(target_data: pd.DataFrame, day_start_hour: int, day_end_hour: int) -> pd.DataFrame:
    target = target_data.copy()
    month_base = pd.DataFrame({"month": MONTH_ORDER})
    month_stats = (
        target.groupby("month", as_index=False)["kw"]
        .agg(最高デマンド値_kW="max", 最低デマンド値_kW="min", 平均デマンド値_kW="mean")
    )
    full_energy = target.groupby("month", as_index=False)["kw"].sum().rename(columns={"kw": "全時間使用量_kWh"})
    full_energy["全時間使用量_kWh"] = full_energy["全時間使用量_kWh"] / 2.0

    day_mask = _time_range_mask(target, int(day_start_hour), int(day_end_hour))
    daytime = target.loc[day_mask].copy()
    day_energy = daytime.groupby("month", as_index=False)["kw"].sum().rename(columns={"kw": "日中使用量_kWh"})
    day_energy["日中使用量_kWh"] = day_energy["日中使用量_kWh"] / 2.0

    month_full_vs_daytime = (
        month_base.merge(month_stats, on="month", how="left")
        .merge(full_energy, on="month", how="left")
        .merge(day_energy, on="month", how="left")
    )
    month_full_vs_daytime["日中利用率_%"] = np.where(
        month_full_vs_daytime["全時間使用量_kWh"] > 0,
        month_full_vs_daytime["日中使用量_kWh"] / month_full_vs_daytime["全時間使用量_kWh"] * 100.0,
        np.nan,
    )
    month_full_vs_daytime["月"] = month_full_vs_daytime["month"].map(lambda x: f"{x}月")
    return month_full_vs_daytime


def estimate_peak_cut(
    current_peak_kw: float,
    target_peak_kw: float,
    unit_price_yen_per_kw_month: float,
    months: int,
    initial_cost_yen: float,
) -> Dict[str, float | None]:
    current = float(max(current_peak_kw, 0.0))
    target = float(max(target_peak_kw, 0.0))
    reduction = max(current - target, 0.0)
    monthly_saving = reduction * max(float(unit_price_yen_per_kw_month), 0.0)
    annual_saving = monthly_saving * max(int(months), 0)
    payback_years = None if annual_saving <= 0 else float(initial_cost_yen) / annual_saving

    return {
        "現在ピーク(kW)": current,
        "目標ピーク(kW)": target,
        "削減量(kW)": reduction,
        "月額削減効果(円)": monthly_saving,
        "年間削減効果(円)": annual_saving,
        "初期費用(円)": float(initial_cost_yen),
        "投資回収年(年)": payback_years,
    }


def run_analysis(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    daytime_enabled: bool,
    day_start_hour: int = 9,
    day_end_hour: int = 18,
) -> AnalysisResult:
    target = _filter_period(df, start_date, end_date)
    if target.empty:
        raise ValueError("選択期間に有効なデータがありません。")

    target = target.copy()
    target["date"] = pd.to_datetime(target["date"]).dt.date
    target["datetime"] = pd.to_datetime(target["datetime"])
    target["month"] = pd.to_datetime(target["date"]).astype("datetime64[ns]").dt.month
    target["weekday"] = pd.to_datetime(target["date"]).astype("datetime64[ns]").dt.weekday
    target["weekday_jp"] = target["weekday"].map(WEEKDAY_JP)

    max_idx = target["kw"].idxmax()
    max_row = target.loc[max_idx]
    summary = {
        "data_days": int(target["date"].nunique()),
        "max_kw": float(target["kw"].max()),
        "max_datetime": pd.to_datetime(max_row["datetime"]),
        "avg_kw": float(target["kw"].mean()),
    }

    daily_peak = (
        target.groupby("date", as_index=False)["kw"]
        .max()
        .rename(columns={"kw": "daily_peak_kw"})
        .sort_values("date")
    )

    avg_curve = (
        target.groupby(["slot", "slot_label"], as_index=False)["kw"]
        .mean()
        .rename(columns={"kw": "avg_kw"})
        .sort_values("slot")
    )

    weekday_curve = (
        target.groupby(["weekday", "weekday_jp", "slot", "slot_label"], as_index=False)["kw"]
        .mean()
        .rename(columns={"kw": "avg_kw"})
        .sort_values(["weekday", "slot"])
    )

    hist_values, bin_edges = np.histogram(target["kw"].to_numpy(), bins=30)
    histogram = pd.DataFrame({"bin_start": bin_edges[:-1], "bin_end": bin_edges[1:], "count": hist_values})

    annual_avg_curve = avg_curve.copy()
    monthly_avg_curve = (
        target.groupby(["month", "slot", "slot_label"], as_index=False)["kw"]
        .mean()
        .rename(columns={"kw": "avg_kw"})
        .sort_values(["month", "slot"])
    )

    monthly_daily_curves = (
        target[["month", "date", "slot", "slot_label", "kw"]]
        .sort_values(["month", "date", "slot"])
        .reset_index(drop=True)
    )

    hourly_data = _build_hourly_data(target)
    hourly_data["month"] = pd.to_datetime(hourly_data["date"]).astype("datetime64[ns]").dt.month
    hourly_data["weekday"] = pd.to_datetime(hourly_data["date"]).astype("datetime64[ns]").dt.weekday

    month_std_30_full = _make_std_pivot(target, "month", MONTH_ORDER, "slot_label", SLOT_LABELS_30, lambda v: f"{v}月")
    weekday_std_30_full = _make_std_pivot(target, "weekday", WEEKDAY_ORDER, "slot_label", SLOT_LABELS_30, lambda v: WEEKDAY_JP[v])
    month_std_60_full = _std30_to_std60(month_std_30_full, SLOT_LABELS_60)
    weekday_std_60_full = _std30_to_std60(weekday_std_30_full, SLOT_LABELS_60)
    month_full_vs_daytime = build_month_full_vs_daytime(target, day_start_hour=day_start_hour, day_end_hour=day_end_hour)

    if daytime_enabled:
        day_mask = _time_range_mask(target, day_start_hour, day_end_hour)
        daytime = target.loc[day_mask].copy()
        if daytime.empty:
            daytime_summary: Optional[Dict[str, object]] = None
            daytime_curve = pd.DataFrame(columns=["slot", "slot_label", "avg_kw"])
        else:
            day_max_idx = daytime["kw"].idxmax()
            day_max_row = daytime.loc[day_max_idx]
            daytime_summary = {
                "avg_kw": float(daytime["kw"].mean()),
                "max_kw": float(daytime["kw"].max()),
                "max_datetime": pd.to_datetime(day_max_row["datetime"]),
            }
            daytime_curve = (
                daytime.groupby(["slot", "slot_label"], as_index=False)["kw"]
                .mean()
                .rename(columns={"kw": "avg_kw"})
                .sort_values("slot")
            )

        day_cols_30 = _daytime_halfhour_labels(day_start_hour, day_end_hour)
        day_cols_60 = _daytime_hour_labels(day_start_hour, day_end_hour)
        month_std_30_daytime = month_std_30_full.reindex(columns=[c for c in day_cols_30 if c in month_std_30_full.columns]).copy()
        weekday_std_30_daytime = weekday_std_30_full.reindex(columns=[c for c in day_cols_30 if c in weekday_std_30_full.columns]).copy()
        month_std_60_daytime = _std30_to_std60(month_std_30_daytime, day_cols_60)
        weekday_std_60_daytime = _std30_to_std60(weekday_std_30_daytime, day_cols_60)
    else:
        daytime_summary = None
        daytime_curve = pd.DataFrame(columns=["slot", "slot_label", "avg_kw"])
        month_std_30_daytime = pd.DataFrame()
        weekday_std_30_daytime = pd.DataFrame()
        month_std_60_daytime = pd.DataFrame()
        weekday_std_60_daytime = pd.DataFrame()

    # Excel「時間別デマンド回数等」定義相当: 日最大デマンドを記録した時刻の回数と、その時刻での最大値
    day_max = target.groupby("date")["kw"].transform("max")
    peak_rows = target[target["kw"] == day_max].copy()

    all_slots = target[["slot", "slot_label"]].drop_duplicates().sort_values("slot")
    slot_count = peak_rows.groupby("slot").size()
    slot_peakmax = peak_rows.groupby("slot")["kw"].max()

    slot_max_and_count = all_slots.copy()
    slot_max_and_count["max_count"] = slot_max_and_count["slot"].map(slot_count).fillna(0).astype(int)
    slot_max_and_count["max_kw"] = slot_max_and_count["slot"].map(slot_peakmax).fillna(0.0)

    demand_range_default = summarize_demand_ranges(target["kw"], bin_width_kw=50.0)

    return AnalysisResult(
        summary=summary,
        daily_peak=daily_peak,
        avg_curve=avg_curve,
        weekday_curve=weekday_curve,
        histogram=histogram,
        target_data=target[["date", "datetime", "month", "weekday", "weekday_jp", "slot", "slot_label", "kw"]].copy(),
        hourly_data=hourly_data[["date", "hour_slot", "hour_label", "kw", "month", "weekday"]].copy(),
        annual_avg_curve=annual_avg_curve,
        monthly_avg_curve=monthly_avg_curve,
        monthly_daily_curves=monthly_daily_curves,
        month_std_30_full=month_std_30_full,
        month_std_30_daytime=month_std_30_daytime,
        month_std_60_full=month_std_60_full,
        month_std_60_daytime=month_std_60_daytime,
        weekday_std_30_full=weekday_std_30_full,
        weekday_std_30_daytime=weekday_std_30_daytime,
        weekday_std_60_full=weekday_std_60_full,
        weekday_std_60_daytime=weekday_std_60_daytime,
        month_full_vs_daytime=month_full_vs_daytime,
        slot_max_and_count=slot_max_and_count,
        demand_range_default=demand_range_default,
        daytime_enabled=daytime_enabled,
        day_start_hour=day_start_hour,
        day_end_hour=day_end_hour,
        daytime_summary=daytime_summary,
        daytime_curve=daytime_curve,
    )
