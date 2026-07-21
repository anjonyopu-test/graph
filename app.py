from __future__ import annotations

from typing import Optional

import streamlit as st

from analysis import run_analysis
from demand_parser import ParsedDemandData, infer_file_kind, list_demand_sheets, parse_uploaded_data
from ui_sections import render_selected_sections


@st.cache_data(show_spinner=False)
def cached_list_sheets(data_bytes: bytes) -> list[str]:
    return list_demand_sheets(data_bytes)


@st.cache_data(show_spinner=False)
def cached_parse_data(data_bytes: bytes, filename: str, input_unit: str, sheet_name: Optional[str]) -> ParsedDemandData:
    return parse_uploaded_data(data_bytes=data_bytes, filename=filename, input_unit=input_unit, sheet_name=sheet_name)


def _reset_result_if_source_changed(key: str) -> None:
    prev = st.session_state.get("source_key")
    if prev != key:
        st.session_state.pop("analysis_result", None)
        st.session_state["source_key"] = key


def main() -> None:
    st.set_page_config(page_title="デマンド分析ツール", layout="wide")
    st.title("デマンド分析ツール")

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None

    st.markdown("### 設定")
    up_col1, up_col2 = st.columns([2, 1])

    with up_col1:
        uploaded_file = st.file_uploader("データファイルを選択（xlsx / csv）", type=["xlsx", "csv"])

    if not uploaded_file:
        st.info("まずファイルをアップロードしてください。")
        return

    data_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    try:
        file_kind = infer_file_kind(filename)
    except ValueError as e:
        st.error(str(e))
        return

    sheet_name = None
    facility_default = filename.rsplit(".", 1)[0]

    with up_col2:
        unit_label = st.selectbox("入力単位", ["kW（30分平均電力）", "kWh（30分エネルギー）"], index=0)
        input_unit = "kwh" if unit_label.startswith("kWh") else "kw"

    if file_kind == "xlsx":
        try:
            sheet_names = cached_list_sheets(data_bytes)
        except Exception as e:
            st.error(f"シート一覧の取得に失敗しました: {e}")
            return

        sheet_name = st.selectbox("シート（施設）を選択", sheet_names)
        facility_default = sheet_name

    facility_name = st.text_input("施設名", value=facility_default)
    if file_kind == "csv" and not facility_name.strip():
        st.warning("csvの場合、施設名の入力は必須です。")
        return

    source_key = f"{filename}|{sheet_name}|{input_unit}"
    _reset_result_if_source_changed(source_key)

    try:
        parsed = cached_parse_data(data_bytes, filename, input_unit, sheet_name)
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        return

    min_date = parsed.data["date"].min()
    max_date = parsed.data["date"].max()

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("期間開始日", value=min_date, min_value=min_date, max_value=max_date)
    with date_col2:
        end_date = st.date_input("期間終了日", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("期間開始日は終了日以前にしてください。")
        return

    run = st.button("分析実行", type="primary")
    if run:
        try:
            result = run_analysis(
                parsed.data,
                start_date,
                end_date,
                daytime_enabled=False,
                day_start_hour=6,
                day_end_hour=18,
            )
            st.session_state["analysis_result"] = {
                "facility_name": facility_name,
                "start_date": start_date,
                "end_date": end_date,
                "daytime_enabled": False,
                "day_start_hour": 6,
                "day_end_hour": 18,
                "result": result,
            }
        except Exception as e:
            st.error(f"分析に失敗しました: {e}")
            return

    state = st.session_state.get("analysis_result")
    if not state:
        st.caption("設定後に「分析実行」を押すと結果を表示します。")
        return

    st.markdown("---")
    st.markdown(f"### 結果: {state['facility_name']}  ({state['start_date']} 〜 {state['end_date']})")

    style_c1, style_c2 = st.columns([2, 1])
    with style_c1:
        st.selectbox("配色テーマ", ["Plotly標準", "Colorblind Safe", "Bright", "Pastel"], index=0, key="line_theme")
    with style_c2:
        st.slider("折れ線の太さ", min_value=1.0, max_value=5.0, value=2.5, step=0.1, key="line_width")

    st.markdown("#### 表示内容")
    cb1, cb2, cb3, cb4 = st.columns(4)
    with cb1:
        show_summary = st.checkbox("サマリー", value=False)
        show_input_heatmap = st.checkbox("入力データ可視化", value=False)
        show_sheet_like = st.checkbox("月毎データ", value=False)
    with cb2:
        show_avg_ext = st.checkbox("グラフ（平均値）", value=False)
        show_monthly_slot_peak = st.checkbox("グラフ（時間帯別ピーク）", value=False)
        show_actual_ext = st.checkbox("グラフ（実データ）", value=False)
        show_monthly_actual = st.checkbox("グラフ（月別実績）", value=False)
        show_weekday_daytime_graph = st.checkbox("曜日別日中デマンド", value=False)
        show_month_daytime_graph = st.checkbox("月別日中デマンド", value=False)
    with cb3:
        show_std = st.checkbox("月別・曜日別 標準偏差", value=False)
        show_range = st.checkbox("デマンドレンジ", value=False)
    with cb4:
        show_peak_cut = st.checkbox("ピークカット目安試算", value=False)
        show_weekday_curve = st.checkbox("曜日別平均負荷曲線", value=False)
        show_slot_max_count = st.checkbox("時間別デマンド回数（時間別最高デマンド値）", value=False)
        show_material_data = st.checkbox("資料作成用データ", value=False)

    render_selected_sections(
        result=state["result"],
        show_summary=show_summary,
        show_weekday_curve=show_weekday_curve,
        show_avg_ext=show_avg_ext,
        show_monthly_slot_peak=show_monthly_slot_peak,
        show_monthly_actual=show_monthly_actual,
        show_weekday_daytime_graph=show_weekday_daytime_graph,
        show_std=show_std,
        show_month_daytime_graph=show_month_daytime_graph,
        show_slot_max_count=show_slot_max_count,
        show_range=show_range,
        show_peak_cut=show_peak_cut,
        show_sheet_like=show_sheet_like,
        show_input_heatmap=show_input_heatmap,
        show_material_data=show_material_data,
        show_actual_ext=show_actual_ext,
        facility_name=state["facility_name"],
        start_date=state["start_date"],
        end_date=state["end_date"],
    )


if __name__ == "__main__":
    main()
