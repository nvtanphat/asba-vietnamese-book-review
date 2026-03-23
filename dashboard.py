from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_REPORT_PATH = Path("experiments/reports/train_scan.json")


DISPLAY_COLUMN_LABELS = {
    "column": "Cột",
    "count": "Số lượng",
    "ratio": "Tỷ lệ (%)",
    "non_empty_count": "Số lượng không rỗng",
    "non_empty_ratio": "Tỷ lệ không rỗng (%)",
    "numeric_like_count": "Số lượng dạng số",
    "numeric_like_ratio": "Tỷ lệ dạng số (%)",
    "type_guess": "Kiểu dự đoán",
    "missing_count": "Số lượng thiếu",
    "missing_ratio": "Tỷ lệ thiếu (%)",
    "non_missing_count": "Số lượng không thiếu",
    "imbalance_ratio": "Tỷ lệ mất cân bằng",
    "value": "Giá trị",
    "text": "Văn bản",
    "token": "Token",
    "emoji": "Emoji",
    "bucket": "Nhóm",
    "issue": "Lỗi",
    "pattern": "Mẫu",
    "min": "Nhỏ nhất",
    "max": "Lớn nhất",
    "mean": "Trung bình",
    "median": "Trung vị",
    "stdev": "Độ lệch chuẩn",
    "p25": "Phân vị 25%",
    "p75": "Phân vị 75%",
}

OVERVIEW_METRIC_LABELS = {
    "non_empty_ratio": "Tỷ lệ không rỗng",
    "numeric_like_ratio": "Tỷ lệ dạng số",
}

TYPE_GUESS_LABELS = {
    "mixed": "Hỗn hợp",
    "numeric_like": "Dạng số",
    "text_like": "Dạng văn bản",
    "unknown": "Không xác định",
}

ENCODING_EXAMPLE_LABELS = {
    "replacement_char": "Ký tự thay thế",
    "zero_width": "Ký tự độ rộng bằng 0",
    "control_char": "Ký tự điều khiển",
    "mojibake_hint": "Dấu hiệu mojibake",
    "fixable_encoding": "Có thể sửa mã hóa",
}

NOISE_EXAMPLE_LABELS = {
    "url": "URL",
    "email": "Email",
    "phone": "Số điện thoại",
    "html": "HTML",
    "markdown_link": "Liên kết Markdown",
    "punct_repeat": "Lặp dấu câu",
    "elongated": "Kéo dài",
    "digit_only": "Chỉ chữ số",
    "symbol_only": "Chỉ ký hiệu",
}


st.set_page_config(
    page_title="Bảng điều khiển kiểm tra dữ liệu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_style() -> None:
    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1600px;
          }
          .app-shell {
            background: linear-gradient(180deg, rgba(14,116,144,0.10) 0%, rgba(244,247,249,1) 18%, rgba(244,247,249,1) 100%);
            padding: 1rem 1.2rem 0.5rem 1.2rem;
            border-radius: 24px;
            margin-bottom: 1rem;
          }
          .hero {
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0f172a 0%, #134e4a 60%, #0f766e 100%);
            color: white;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
          }
          .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.05;
          }
          .hero p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
          }
          .section-card {
            background: white;
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
          }
          .small-label {
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.3rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_report_from_path(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_report(source: str | None, uploaded: Any | None) -> tuple[dict[str, Any] | None, str]:
    if uploaded is not None:
        return json.loads(uploaded.read().decode("utf-8")), uploaded.name
    if source:
        path = Path(source)
        if path.exists():
            return load_report_from_path(path), str(path)
    return None, source or ""


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def translate_df_columns(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={column: label_map.get(column, column) for column in df.columns})


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def as_dataframe(items: list[dict[str, Any]], preferred_name: str = "item") -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=[preferred_name, "count"])
    df = pd.DataFrame(items).copy()
    if "count" not in df.columns:
        df["count"] = 1
    renamed = False
    for candidate in (preferred_name, "token", "emoji", "text", "name", "value", "column"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: preferred_name})
            renamed = True
            break
    if not renamed and df.columns.tolist():
        df = df.rename(columns={df.columns[0]: preferred_name})
    if preferred_name not in df.columns:
        df[preferred_name] = ""
    ordered_cols = [preferred_name] + [c for c in df.columns if c not in {preferred_name, "count"}] + ["count"]
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    return df[ordered_cols]


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty:
        st.info("Không có dữ liệu để vẽ.")
        return
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color if color and color in df.columns else None,
        title=title,
        text_auto=".2f" if pd.api.types.is_numeric_dtype(df[y]) else True,
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_table(title: str, df: pd.DataFrame, height: int = 360) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.info("Không có dữ liệu.")
        return
    st.dataframe(df, use_container_width=True, height=height)


def render_overview(check: dict[str, Any]) -> None:
    section_title("Tổng quan", "Tổng quan dữ liệu và độ đầy cột")

    row_count = check["row_count"]
    col_count = check["column_count"]
    columns = check["columns"]
    non_empty_counts = check["non_empty_counts"]
    numeric_like_counts = check["numeric_like_counts"]
    type_guess = check["column_type_guess"]

    metric_row(
        [
            ("Số dòng", f"{row_count:,}"),
            ("Số cột", f"{col_count:,}"),
            ("Cột văn bản", str(check["text_column"])),
            ("Số kiểu phát hiện", f"{len(set(type_guess.values()))}"),
        ]
    )

    df = pd.DataFrame(
        {
            "column": columns,
            "non_empty_count": [non_empty_counts.get(c, 0) for c in columns],
            "non_empty_ratio": [round(non_empty_counts.get(c, 0) * 100 / row_count, 2) for c in columns],
            "numeric_like_count": [numeric_like_counts.get(c, 0) for c in columns],
            "numeric_like_ratio": [round(numeric_like_counts.get(c, 0) * 100 / row_count, 2) for c in columns],
            "type_guess": [type_guess.get(c, "unknown") for c in columns],
        }
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        chart_df = df.melt(
            id_vars=["column"],
            value_vars=["non_empty_ratio", "numeric_like_ratio"],
            var_name="metric",
            value_name="ratio",
        )
        chart_df = chart_df.rename(columns={"column": "Cột", "metric": "Chỉ số", "ratio": "Tỷ lệ (%)"})
        chart_df["Chỉ số"] = chart_df["Chỉ số"].map(OVERVIEW_METRIC_LABELS).fillna(chart_df["Chỉ số"])
        fig = px.bar(
            chart_df,
            x="Cột",
            y="Tỷ lệ (%)",
            color="Chỉ số",
            barmode="group",
            title="Tỷ lệ không rỗng và dạng số theo cột",
            color_discrete_sequence=["#0f766e", "#ea580c"],
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        type_df = df["type_guess"].value_counts().reset_index()
        type_df.columns = ["type_guess", "count"]
        type_df["type_guess"] = type_df["type_guess"].map(TYPE_GUESS_LABELS).fillna(type_df["type_guess"])
        type_df = type_df.rename(columns={"type_guess": "Kiểu dự đoán", "count": "Số lượng"})
        fig = px.pie(
            type_df,
            values="Số lượng",
            names="Kiểu dự đoán",
            title="Phân loại cột",
            color_discrete_sequence=["#0f766e", "#14b8a6", "#f59e0b", "#94a3b8"],
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    display_df = translate_df_columns(df, DISPLAY_COLUMN_LABELS)
    display_df["Kiểu dự đoán"] = display_df["Kiểu dự đoán"].map(TYPE_GUESS_LABELS).fillna(display_df["Kiểu dự đoán"])
    show_table("Chi tiết cột", display_df)


def render_missing(check: dict[str, Any]) -> None:
    section_title("Giá trị thiếu", "Tỷ lệ giá trị rỗng theo cột")

    metric_row(
        [
            ("Dòng có thiếu", f"{check['rows_with_at_least_one_missing']:,}"),
            ("Tỷ lệ dòng có thiếu", f"{check['rows_with_at_least_one_missing_ratio']:.2f}%"),
        ]
    )

    per_column = check["per_column"]
    df = pd.DataFrame.from_dict(per_column, orient="index").reset_index().rename(columns={"index": "column"})
    df = df.sort_values("missing_ratio", ascending=False)
    bar_chart(df.head(25), x="column", y="missing_ratio", title="Các cột có tỷ lệ thiếu cao nhất", color=None)
    show_table("Thiếu theo cột", translate_df_columns(df, DISPLAY_COLUMN_LABELS))

    top_missing = pd.DataFrame(check["top_missing_columns"])
    if not top_missing.empty:
        show_table("Các cột thiếu nhiều nhất", translate_df_columns(top_missing, DISPLAY_COLUMN_LABELS))


def render_length(check: dict[str, Any]) -> None:
    section_title("Độ dài", "Phân bố độ dài nội dung")
    ls = check["length_summary"]
    ws = check["word_summary"]

    metric_row(
        [
            ("Cột văn bản", str(check["text_column"])),
            ("Ngắn hơn ngưỡng tối thiểu", f"{check['shorter_than_min_length']:,}"),
            ("Tỷ lệ ngắn", f"{check['shorter_than_min_length_ratio']:.2f}%"),
            ("Độ dài trung bình", f"{ls['mean']:.2f}" if ls["mean"] is not None else "không có"),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Thống kê độ dài**")
        st.dataframe(translate_df_columns(pd.DataFrame([ls]), DISPLAY_COLUMN_LABELS), use_container_width=True, height=240)
    with right:
        st.markdown("**Thống kê số từ**")
        st.dataframe(translate_df_columns(pd.DataFrame([ws]), DISPLAY_COLUMN_LABELS), use_container_width=True, height=240)

    bucket_df = pd.DataFrame(
        [{"bucket": bucket, "count": count} for bucket, count in check["length_buckets"].items()]
    )
    bar_chart(bucket_df, x="bucket", y="count", title="Phân bố nhóm độ dài")


def render_text_examples(title: str, examples: dict[str, list[str]], key_labels: dict[str, str] | None = None) -> None:
    with st.expander(title, expanded=False):
        if not examples:
            st.info("Không có mẫu.")
            return
        for key, values in examples.items():
            display_key = key_labels.get(key, key) if key_labels else key
            st.markdown(f"**{display_key}**")
            if values:
                for value in values:
                    st.code(str(value), language="text")
            else:
                st.caption("Không có mẫu.")


def render_encoding(check: dict[str, Any]) -> None:
    section_title("Mã hóa & Unicode", "Vấn đề Unicode, mojibake và ký tự điều khiển")
    metric_row(
        [
            ("Dòng có lỗi", f"{check['rows_with_any_issue']:,}"),
            ("Tỷ lệ dòng có lỗi", f"{check['rows_with_any_issue_ratio']:.2f}%"),
        ]
    )

    df = (
        pd.DataFrame([{ "issue": key, "count": count } for key, count in check["issue_counts"].items()])
        .sort_values("count", ascending=False)
        if check["issue_counts"]
        else pd.DataFrame(columns=["issue", "count"])
    )
    if not df.empty:
        df["issue"] = df["issue"].map(ENCODING_EXAMPLE_LABELS).fillna(df["issue"])
        df = df.rename(columns={"issue": "Loại lỗi", "count": "Số lượng"})
    bar_chart(df, x="Loại lỗi", y="Số lượng", title="Số lượng lỗi mã hóa")
    render_text_examples("Ví dụ mã hóa", check["examples"], ENCODING_EXAMPLE_LABELS)


def render_noise(check: dict[str, Any]) -> None:
    section_title("Nhiễu", "URL, email, số điện thoại, HTML, dấu câu lặp, số/ký hiệu")
    metric_row(
        [
            ("Dòng có nhiễu", f"{check['rows_with_any_noise']:,}"),
            ("Tỷ lệ dòng có nhiễu", f"{check['rows_with_any_noise_ratio']:.2f}%"),
        ]
    )

    df = (
        pd.DataFrame([{ "pattern": key, "count": count } for key, count in check["pattern_counts"].items()])
        .sort_values("count", ascending=False)
        if check["pattern_counts"]
        else pd.DataFrame(columns=["pattern", "count"])
    )
    if not df.empty:
        df["pattern"] = df["pattern"].map(NOISE_EXAMPLE_LABELS).fillna(df["pattern"])
        df = df.rename(columns={"pattern": "Mẫu nhiễu", "count": "Số lượng"})
    bar_chart(df, x="Mẫu nhiễu", y="Số lượng", title="Số lượng mẫu nhiễu")
    render_text_examples("Ví dụ nhiễu", check["examples"], NOISE_EXAMPLE_LABELS)


def render_emoji(check: dict[str, Any]) -> None:
    section_title("Emoji", "Tất cả emoji xuất hiện trong báo cáo")
    metric_row(
        [
            ("Dòng có emoji", f"{check['rows_with_emoji']:,}"),
            ("Tỷ lệ emoji", f"{check['rows_with_emoji_ratio']:.2f}%"),
            ("Tổng emoji", f"{check['emoji_total']:,}"),
            ("Emoji duy nhất", f"{len(check['all_emojis']):,}"),
        ]
    )

    df = as_dataframe(check["all_emojis"], preferred_name="emoji")
    display_df = translate_df_columns(df, DISPLAY_COLUMN_LABELS)
    bar_chart(display_df.head(40), x="Emoji", y="Số lượng", title="Emoji xuất hiện nhiều nhất")
    st.caption("Danh sách đầy đủ nằm bên dưới.")
    show_table("Tất cả emoji", display_df, height=420)

    samples = check.get("emoji_samples", [])
    with st.expander("Mẫu emoji", expanded=False):
        if samples:
            st.code(" ".join(samples), language="text")
        else:
            st.info("Không có mẫu emoji.")


def render_vocab(check: dict[str, Any]) -> None:
    section_title("Từ vựng", "Teencode, từ bị kéo dài, token nghi vấn")
    metric_row(
        [
            ("Dòng đáng ngờ", f"{check['rows_with_suspicious_vocab']:,}"),
            ("Tỷ lệ đáng ngờ", f"{check['rows_with_suspicious_vocab_ratio']:.2f}%"),
            ("Tổng token", f"{check['token_stats'].get('total_tokens', 0):,}"),
            ("Token duy nhất", f"{check['token_stats'].get('unique_tokens', 0):,}"),
        ]
    )

    groups = [
        ("Token kiểu teencode", check.get("teencode_like_tokens", [])),
        ("Token kéo dài", check.get("elongated_tokens", [])),
        ("Token ASCII không dấu", check.get("accentless_ascii_tokens", [])),
        ("Token chữ-số trộn", check.get("mixed_alnum_tokens", [])),
        ("Từ có thể sai chính tả", check.get("possible_misspellings", [])),
    ]

    tab_objs = st.tabs([title for title, _ in groups])
    for tab, (title, items) in zip(tab_objs, groups):
        with tab:
            df = as_dataframe(items, preferred_name="token")
            display_df = translate_df_columns(df, DISPLAY_COLUMN_LABELS)
            bar_chart(display_df.head(40), x="Token", y="Số lượng", title=title)
            show_table(f"{title} - danh sách đầy đủ", display_df, height=360)


def render_duplicates(check: dict[str, Any]) -> None:
    section_title("Trùng lặp", "Trùng nguyên văn và trùng sau chuẩn hóa")
    metric_row(
        [
            ("Dòng trùng nguyên văn", f"{check['exact_duplicate_rows']:,}"),
            ("Trùng sau chuẩn hóa", f"{check['normalized_duplicate_texts']:,}"),
            ("Tỷ lệ trùng nguyên văn", f"{check['exact_duplicate_rows_ratio']:.2f}%"),
            ("Tỷ lệ trùng sau chuẩn hóa", f"{check['normalized_duplicate_texts_ratio']:.2f}%"),
        ]
    )

    df = as_dataframe(check.get("top_duplicate_texts", []), preferred_name="text")
    show_table("Các văn bản trùng nhiều nhất", translate_df_columns(df, DISPLAY_COLUMN_LABELS), height=320)


def render_labels(check: dict[str, Any]) -> None:
    section_title("Nhãn", "Phân bố các cột nhãn")
    label_columns = check.get("label_columns", [])
    summary_rows: list[dict[str, Any]] = []

    for column in label_columns:
        stats = check["columns"].get(column, {})
        summary_rows.append(
            {
                "column": column,
                "missing_count": stats.get("missing_count", 0),
                "missing_ratio": stats.get("missing_ratio", 0.0),
                "non_missing_count": stats.get("non_missing_count", 0),
                "imbalance_ratio": stats.get("imbalance_ratio", None),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("missing_ratio", ascending=False) if summary_rows else pd.DataFrame()
    show_table("Tổng hợp nhãn", translate_df_columns(summary_df, DISPLAY_COLUMN_LABELS), height=260)

    for column in label_columns:
        stats = check["columns"].get(column, {})
        with st.expander(f"Nhãn: {column}", expanded=False):
            metrics = st.columns(4)
            metrics[0].metric("Thiếu", f"{stats.get('missing_count', 0):,}")
            metrics[1].metric("Tỷ lệ thiếu", f"{stats.get('missing_ratio', 0.0):.2f}%")
            metrics[2].metric("Không thiếu", f"{stats.get('non_missing_count', 0):,}")
            metrics[3].metric("Mất cân bằng", str(stats.get("imbalance_ratio")))

            vc_df = as_dataframe(stats.get("value_counts", []), preferred_name="value")
            display_vc_df = translate_df_columns(vc_df, DISPLAY_COLUMN_LABELS)
            bar_chart(display_vc_df, x="Giá trị", y="Số lượng", title=f"Thống kê giá trị - {column}")
            show_table(f"Thống kê giá trị - {column}", display_vc_df, height=260)


def render_raw(report: dict[str, Any]) -> None:
    section_title("JSON gốc", "Xem report gốc dưới dạng JSON")
    st.json(report)


def sidebar(report_path_default: str) -> tuple[dict[str, Any] | None, str]:
    st.sidebar.title("Bảng điều khiển")
    st.sidebar.caption("Đọc report JSON của tập train và trực quan hóa toàn bộ các kiểm tra.")
    path = st.sidebar.text_input("Đường dẫn report", value=report_path_default)
    uploaded = st.sidebar.file_uploader("Hoặc tải lên file report JSON", type=["json"])

    report, source_name = load_report(path, uploaded)
    if report is None:
        st.sidebar.error("Không tìm thấy file report JSON.")
        return None, source_name

    st.sidebar.success(f"Đã tải: {source_name}")
    st.sidebar.download_button(
        "Tải xuống report JSON",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name=Path(source_name).name if source_name else "train_scan.json",
        mime="application/json",
    )
    return report, source_name


def main() -> None:
    apply_style()
    report, source_name = sidebar(str(DEFAULT_REPORT_PATH))
    if report is None:
        st.stop()

    metadata = report.get("metadata", {})
    checks = report.get("checks", {})

    st.markdown(
        """
        <div class="hero">
          <h1>Bảng điều khiển kiểm tra dữ liệu</h1>
          <p>Trực quan hóa toàn bộ report từ bước quét tiền xử lý, không bỏ sót kiểm tra nào.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Nguồn: {source_name}")

    metric_row(
        [
            ("Số dòng", f"{metadata.get('row_count', 0):,}"),
            ("Số cột", f"{metadata.get('column_count', 0):,}"),
            ("Cột văn bản", str(metadata.get("text_column", ""))),
            ("Thời điểm tạo", str(metadata.get("generated_at", ""))),
        ]
    )

    tabs = st.tabs(
        [
            "Tổng quan",
            "Thiếu & độ dài",
            "Mã hóa & nhiễu",
            "Emoji & từ vựng",
            "Trùng lặp & nhãn",
            "JSON gốc",
        ]
    )

    with tabs[0]:
        render_overview(checks["overview"])
    with tabs[1]:
        render_missing(checks["missing_values"])
        render_length(checks["length"])
    with tabs[2]:
        render_encoding(checks["encoding"])
        render_noise(checks["noise_patterns"])
    with tabs[3]:
        render_emoji(checks["emoji"])
        render_vocab(checks["vocab"])
    with tabs[4]:
        render_duplicates(checks["duplicates"])
        render_labels(checks["labels"])
    with tabs[5]:
        render_raw(report)


if __name__ == "__main__":
    main()
