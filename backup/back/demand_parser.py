from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

SLOT_MINUTES = 30
SLOTS_PER_DAY = 48
INTERNAL_SLOT_LABELS = [f"{(i * SLOT_MINUTES) // 60:02d}:{(i * SLOT_MINUTES) % 60:02d}" for i in range(SLOTS_PER_DAY)]


@dataclass(frozen=True)
class ParsedDemandData:
    data: pd.DataFrame
    facility_name_default: str


def infer_file_kind(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".xlsx"):
        return "xlsx"
    if lower.endswith(".csv"):
        return "csv"
    raise ValueError("対応形式は xlsx / csv のみです。")


def _normalize_text(value: object) -> str:
    return str(value).strip().replace("\n", " ")


def _parse_time_to_minutes(label: str) -> Optional[int]:
    text = _normalize_text(label)
    if not text:
        return None

    simple_match = re.match(r"^(\d{1,2})\s*:\s*(\d{2})(?::\d{2})?$", text)
    if simple_match:
        hour = int(simple_match.group(1))
        minute = int(simple_match.group(2))
        if hour == 24 and minute == 0:
            return 0
        if 0 <= hour <= 23 and minute in (0, 30):
            return hour * 60 + minute
        return None

    jp_match = re.match(r"^(\d{1,2})\s*時(?:\s*(\d{1,2})\s*分?)?$", text)
    if jp_match:
        hour = int(jp_match.group(1))
        minute_text = jp_match.group(2)
        minute = int(minute_text) if minute_text else 0
        if hour == 24 and minute == 0:
            return 0
        if 0 <= hour <= 23 and minute in (0, 30):
            return hour * 60 + minute
        return None

    return None


def _ending_label_to_internal_slot_index(minutes: int) -> int:
    return ((minutes - SLOT_MINUTES) % (24 * 60)) // SLOT_MINUTES


def _detect_date_column(df: pd.DataFrame) -> str:
    candidates = [str(c) for c in df.columns]
    for col in candidates:
        norm = _normalize_text(col).lower()
        if any(key in norm for key in ("日付", "date", "年月日")):
            return col
    return candidates[0]


def _promote_header_row_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    max_check = min(8, len(work))
    for row_idx in range(max_check):
        row_values = [_normalize_text(v) for v in work.iloc[row_idx].tolist()]
        time_like = sum(1 for v in row_values if _parse_time_to_minutes(v) is not None)
        has_date_word = any(any(k in v.lower() for k in ("日付", "date", "年月日")) for v in row_values)
        if time_like >= 20 and has_date_word:
            work.columns = row_values
            work = work.iloc[row_idx + 1 :].reset_index(drop=True)
            return work
    return work


def _extract_time_columns(df: pd.DataFrame, date_col: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for col in df.columns:
        if str(col) == str(date_col):
            continue
        minutes = _parse_time_to_minutes(str(col))
        if minutes is None:
            continue
        mapping[str(col)] = _ending_label_to_internal_slot_index(minutes)
    return mapping


def _normalize_frame(df: pd.DataFrame, input_unit: str) -> pd.DataFrame:
    work = _promote_header_row_if_needed(df)
    work.columns = [str(c).strip() for c in work.columns]

    date_col = _detect_date_column(work)
    slot_mapping = _extract_time_columns(work, date_col)
    if not slot_mapping:
        raise ValueError("30分時間列を認識できませんでした。時刻列(例: 00:30, 0時30分)を確認してください。")

    melted = work.melt(id_vars=[date_col], value_vars=list(slot_mapping.keys()), var_name="input_slot", value_name="value")
    melted["date"] = pd.to_datetime(melted[date_col], errors="coerce", format="mixed").dt.date
    melted["kw"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["date", "kw"]).copy()

    if input_unit == "kwh":
        melted["kw"] = melted["kw"] * 2.0

    melted["slot"] = melted["input_slot"].map(slot_mapping).astype(int)
    melted["slot_label"] = melted["slot"].map(lambda s: INTERNAL_SLOT_LABELS[s])
    melted["datetime"] = pd.to_datetime(melted["date"]) + pd.to_timedelta(melted["slot"] * SLOT_MINUTES, unit="m")

    output = melted[["date", "slot", "slot_label", "datetime", "kw"]].sort_values(["date", "slot"]).reset_index(drop=True)
    return output


def _read_csv(data_bytes: bytes) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp932", "utf-8"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(BytesIO(data_bytes), encoding=enc)
        except Exception as e:
            last_error = e
    raise ValueError(f"CSVの読み込みに失敗しました: {last_error}")


def _read_excel_sheet(data_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(data_bytes), sheet_name=sheet_name)


def list_excel_sheets(data_bytes: bytes) -> List[str]:
    with pd.ExcelFile(BytesIO(data_bytes)) as xls:
        return xls.sheet_names


def list_demand_sheets(data_bytes: bytes) -> List[str]:
    valid: List[str] = []
    all_sheets: List[str] = []
    with pd.ExcelFile(BytesIO(data_bytes)) as xls:
        all_sheets = list(xls.sheet_names)
        for name in xls.sheet_names:
            try:
                raw = pd.read_excel(xls, sheet_name=name)
                normalized = _normalize_frame(raw, input_unit="kw")
                has_enough_slots = normalized["slot"].nunique() >= 24
                has_dates = normalized["date"].nunique() >= 1
                if (len(normalized) >= 48) and has_enough_slots and has_dates:
                    valid.append(name)
            except Exception:
                continue
    if valid:
        return valid

    excluded_keywords = [
        "グラフ",
        "標準偏差",
        "レンジ",
        "試算",
        "(月毎データ)",
        "(日)",
        "(月)",
        "(火)",
        "(水)",
        "(木)",
        "(金)",
        "(土)",
    ]
    fallback = [s for s in all_sheets if not any(k in s for k in excluded_keywords)]
    return fallback if fallback else all_sheets


def parse_uploaded_data(data_bytes: bytes, filename: str, input_unit: str, sheet_name: Optional[str] = None) -> ParsedDemandData:
    kind = infer_file_kind(filename)

    if kind == "xlsx":
        if not sheet_name:
            raise ValueError("xlsxの場合はシート選択が必要です。")
        raw = _read_excel_sheet(data_bytes, sheet_name)
        default_name = sheet_name
    else:
        raw = _read_csv(data_bytes)
        default_name = filename.rsplit(".", 1)[0]

    normalized = _normalize_frame(raw, input_unit=input_unit)
    if normalized.empty:
        raise ValueError("有効な30分データが見つかりませんでした。シート内容を確認してください。")
    return ParsedDemandData(data=normalized, facility_name_default=default_name)
