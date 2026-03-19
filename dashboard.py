from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_REPORT_PATH = Path("experiments/reports/train_scan.json")


st.set_page_config(
    page_title="Data Scan Dashboard",
    page_icon="Dashboard",
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
    section_title("Overview", "Tổng quan dữ liệu và độ đầy cột")

    row_count = check["row_count"]
    col_count = check["column_count"]
    columns = check["columns"]
    non_empty_counts = check["non_empty_counts"]
    numeric_like_counts = check["numeric_like_counts"]
    type_guess = check["column_type_guess"]

    metric_row(
        [
            ("Rows", f"{row_count:,}"),
            ("Columns", f"{col_count:,}"),
            ("Text column", str(check["text_column"])),
            ("Detected types", f"{len(set(type_guess.values()))}"),
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
        fig = px.bar(
            chart_df,
            x="column",
            y="ratio",
            color="metric",
            barmode="group",
            title="Tỷ lệ non-empty và numeric-like theo cột",
            color_discrete_sequence=["#0f766e", "#ea580c"],
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        type_df = df["type_guess"].value_counts().reset_index()
        type_df.columns = ["type_guess", "count"]
        fig = px.pie(
            type_df,
            values="count",
            names="type_guess",
            title="Phân loại cột",
            color_discrete_sequence=["#0f766e", "#14b8a6", "#f59e0b", "#94a3b8"],
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    show_table("Chi tiết cột", df)


def render_missing(check: dict[str, Any]) -> None:
    section_title("Missing Values", "Tỷ lệ giá trị rỗng theo cột")

    metric_row(
        [
            ("Rows with missing", f"{check['rows_with_at_least_one_missing']:,}"),
            ("Missing ratio", f"{check['rows_with_at_least_one_missing_ratio']:.2f}%"),
        ]
    )

    per_column = check["per_column"]
    df = pd.DataFrame.from_dict(per_column, orient="index").reset_index().rename(columns={"index": "column"})
    df = df.sort_values("missing_ratio", ascending=False)
    bar_chart(df.head(25), x="column", y="missing_ratio", title="Top cột missing ratio", color=None)
    show_table("Missing theo cột", df)

    top_missing = pd.DataFrame(check["top_missing_columns"])
    if not top_missing.empty:
        show_table("Top cột thiếu nhiều nhất", top_missing)


def render_length(check: dict[str, Any]) -> None:
    section_title("Length", "Phân bố độ dài nội dung")
    ls = check["length_summary"]
    ws = check["word_summary"]

    metric_row(
        [
            ("Text column", str(check["text_column"])),
            ("Shorter than min", f"{check['shorter_than_min_length']:,}"),
            ("Short ratio", f"{check['shorter_than_min_length_ratio']:.2f}%"),
            ("Length mean", f"{ls['mean']:.2f}" if ls["mean"] is not None else "n/a"),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Length summary**")
        st.dataframe(pd.DataFrame([ls]), use_container_width=True, height=240)
    with right:
        st.markdown("**Word summary**")
        st.dataframe(pd.DataFrame([ws]), use_container_width=True, height=240)

    bucket_df = pd.DataFrame(
        [{"bucket": bucket, "count": count} for bucket, count in check["length_buckets"].items()]
    )
    bar_chart(bucket_df, x="bucket", y="count", title="Buckets độ dài")


def render_text_examples(title: str, examples: dict[str, list[str]]) -> None:
    with st.expander(title, expanded=False):
        if not examples:
            st.info("Không có mẫu.")
            return
        for key, values in examples.items():
            st.markdown(f"**{key}**")
            if values:
                for value in values:
                    st.code(str(value), language="text")
            else:
                st.caption("Không có mẫu.")


def render_encoding(check: dict[str, Any]) -> None:
    section_title("Encoding", "Vấn đề Unicode, mojibake, control characters")
    metric_row(
        [
            ("Any issue rows", f"{check['rows_with_any_issue']:,}"),
            ("Issue ratio", f"{check['rows_with_any_issue_ratio']:.2f}%"),
        ]
    )

    df = pd.DataFrame(
        [{"issue": key, "count": count} for key, count in check["issue_counts"].items()]
    ).sort_values("count", ascending=False) if check["issue_counts"] else pd.DataFrame(columns=["issue", "count"])
    bar_chart(df, x="issue", y="count", title="Encoding issue counts")
    render_text_examples("Encoding examples", check["examples"])


def render_noise(check: dict[str, Any]) -> None:
    section_title("Noise", "URL, email, phone, html, dấu câu lặp, số/ký hiệu")
    metric_row(
        [
            ("Any noise rows", f"{check['rows_with_any_noise']:,}"),
            ("Noise ratio", f"{check['rows_with_any_noise_ratio']:.2f}%"),
        ]
    )

    df = pd.DataFrame(
        [{"pattern": key, "count": count} for key, count in check["pattern_counts"].items()]
    ).sort_values("count", ascending=False) if check["pattern_counts"] else pd.DataFrame(columns=["pattern", "count"])
    bar_chart(df, x="pattern", y="count", title="Noise pattern counts")
    render_text_examples("Noise examples", check["examples"])


def render_emoji(check: dict[str, Any]) -> None:
    section_title("Emoji", "Tất cả emoji xuất hiện trong report")
    metric_row(
        [
            ("Rows with emoji", f"{check['rows_with_emoji']:,}"),
            ("Emoji ratio", f"{check['rows_with_emoji_ratio']:.2f}%"),
            ("Total emoji", f"{check['emoji_total']:,}"),
            ("Unique emoji", f"{len(check['all_emojis']):,}"),
        ]
    )

    df = as_dataframe(check["all_emojis"], preferred_name="emoji")
    bar_chart(df.head(40), x="emoji", y="count", title="Top emoji")
    st.caption("Danh sách đầy đủ bên dưới.")
    show_table("All emoji", df, height=420)

    samples = check.get("emoji_samples", [])
    with st.expander("Emoji samples", expanded=False):
        if samples:
            st.code(" ".join(samples), language="text")
        else:
            st.info("Không có sample emoji.")


def render_vocab(check: dict[str, Any]) -> None:
    section_title("Vocabulary", "Teencode, từ bị kéo dài, token nghi vấn")
    metric_row(
        [
            ("Suspicious rows", f"{check['rows_with_suspicious_vocab']:,}"),
            ("Suspicious ratio", f"{check['rows_with_suspicious_vocab_ratio']:.2f}%"),
            ("Total tokens", f"{check['token_stats'].get('total_tokens', 0):,}"),
            ("Unique tokens", f"{check['token_stats'].get('unique_tokens', 0):,}"),
        ]
    )

    groups = [
        ("Teencode-like tokens", check.get("teencode_like_tokens", [])),
        ("Elongated tokens", check.get("elongated_tokens", [])),
        ("Accentless ascii tokens", check.get("accentless_ascii_tokens", [])),
        ("Mixed alnum tokens", check.get("mixed_alnum_tokens", [])),
        ("Possible misspellings", check.get("possible_misspellings", [])),
    ]

    tab_objs = st.tabs([title for title, _ in groups])
    for tab, (title, items) in zip(tab_objs, groups):
        with tab:
            df = as_dataframe(items, preferred_name="token")
            bar_chart(df.head(40), x="token", y="count", title=title)
            show_table(f"{title} - full list", df, height=360)


def render_duplicates(check: dict[str, Any]) -> None:
    section_title("Duplicates", "Trùng nguyên văn và trùng sau chuẩn hóa")
    metric_row(
        [
            ("Exact duplicate rows", f"{check['exact_duplicate_rows']:,}"),
            ("Normalized duplicates", f"{check['normalized_duplicate_texts']:,}"),
            ("Exact ratio", f"{check['exact_duplicate_rows_ratio']:.2f}%"),
            ("Normalized ratio", f"{check['normalized_duplicate_texts_ratio']:.2f}%"),
        ]
    )

    df = as_dataframe(check.get("top_duplicate_texts", []), preferred_name="text")
    show_table("Top duplicate texts", df, height=320)


def render_labels(check: dict[str, Any]) -> None:
    section_title("Labels", "Phân bố các cột nhãn")
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
    show_table("Label summary", summary_df, height=260)

    for column in label_columns:
        stats = check["columns"].get(column, {})
        with st.expander(f"Label: {column}", expanded=False):
            metrics = st.columns(4)
            metrics[0].metric("Missing", f"{stats.get('missing_count', 0):,}")
            metrics[1].metric("Missing ratio", f"{stats.get('missing_ratio', 0.0):.2f}%")
            metrics[2].metric("Non-missing", f"{stats.get('non_missing_count', 0):,}")
            metrics[3].metric("Imbalance", str(stats.get("imbalance_ratio")))

            vc_df = as_dataframe(stats.get("value_counts", []), preferred_name="value")
            bar_chart(vc_df, x="value", y="count", title=f"Value counts - {column}")
            show_table(f"Value counts - {column}", vc_df, height=260)


def render_raw(report: dict[str, Any]) -> None:
    section_title("Raw JSON", "Xem nguyên report gốc")
    st.json(report)


def sidebar(report_path_default: str) -> tuple[dict[str, Any] | None, str]:
    st.sidebar.title("Dashboard")
    st.sidebar.caption("Đọc report JSON của train split và trực quan hóa toàn bộ check.")
    path = st.sidebar.text_input("Report path", value=report_path_default)
    uploaded = st.sidebar.file_uploader("Or upload report JSON", type=["json"])

    report, source_name = load_report(path, uploaded)
    if report is None:
        st.sidebar.error("Không tìm thấy report JSON.")
        return None, source_name

    st.sidebar.success(f"Loaded: {source_name}")
    st.sidebar.download_button(
        "Download report JSON",
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
          <h1>Data Scan Dashboard</h1>
          <p>Trực quan hóa toàn bộ report từ preprocessing scan, không bỏ sót check nào.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Source: {source_name}")

    metric_row(
        [
            ("Rows", f"{metadata.get('row_count', 0):,}"),
            ("Columns", f"{metadata.get('column_count', 0):,}"),
            ("Text column", str(metadata.get("text_column", ""))),
            ("Generated at", str(metadata.get("generated_at", ""))),
        ]
    )

    tabs = st.tabs(
        [
            "Summary",
            "Missing & Length",
            "Encoding & Noise",
            "Emoji & Vocab",
            "Duplicates & Labels",
            "Raw JSON",
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
