import html
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.predictor import ABSAPredictor, ASPECT_COLS, ASPECT_NAMES, MODEL_VARIANTS
from src.ui.styles import CSS


st.set_page_config(page_title="ABSA Dashboard", layout="wide")
if CSS:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_variant: str):
    return ABSAPredictor(model_variant=model_variant)


MODEL_VARIANT_ORDER = ["phobert", "bilstm-phobert"]
MODEL_LABEL_TO_KEY = {MODEL_VARIANTS[key]["label"]: key for key in MODEL_VARIANT_ORDER}
MODEL_KEY_TO_LABEL = {key: MODEL_VARIANTS[key]["label"] for key in MODEL_VARIANT_ORDER}


SENTIMENT_META = {
    0: {"label": "Tieu cuc", "color": "#f87171"},
    1: {"label": "Trung lap", "color": "#fbbf24"},
    2: {"label": "Tich cuc", "color": "#4ade80"},
}


def fmt_pct(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%"


def stat_card(label: str, value: str, hint: str) -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-label">{html.escape(label)}</div>
        <div class="stat-value">{html.escape(value)}</div>
        <div class="stat-hint">{html.escape(hint)}</div>
    </div>
    """


def dominant_sentiment(sentiment_probs: list[float]) -> tuple[int, float]:
    idx = max(range(len(sentiment_probs)), key=lambda i: sentiment_probs[i])
    return idx, sentiment_probs[idx]


def build_aspect_rows(result: dict) -> list[dict]:
    rows = []
    probs = result["aspect_probs"]
    for i, col_key in enumerate(ASPECT_COLS):
        presence = float(probs[col_key]["presence"])
        is_present = result["aspects"][col_key] != -1
        if is_present:
            sentiment_probs = [float(x) for x in probs[col_key]["sentiment"]]
            level_idx, level_conf = dominant_sentiment(sentiment_probs)
            level_label = SENTIMENT_META[level_idx]["label"]
        else:
            sentiment_probs = [0.0, 0.0, 0.0]
            level_conf = 0.0
            level_label = "Khong ap dung"

        rows.append(
            {
                "aspect": ASPECT_NAMES[i],
                "presence": presence,
                "mentioned": is_present,
                "level_label": level_label,
                "level_conf": float(level_conf),
                "neg": sentiment_probs[0],
                "neu": sentiment_probs[1],
                "pos": sentiment_probs[2],
            }
        )
    return sorted(rows, key=lambda x: x["presence"], reverse=True)


def make_overall_prob_figure(overall_probs: list[float]) -> go.Figure:
    labels = [SENTIMENT_META[i]["label"] for i in range(3)]
    colors = [SENTIMENT_META[i]["color"] for i in range(3)]
    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=overall_probs,
            marker_color=colors,
            text=[fmt_pct(v) for v in overall_probs],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        height=145,
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis=dict(title=""),
        showlegend=False,
    )
    return fig


def make_aspect_presence_figure(aspect_rows: list[dict]) -> go.Figure:
    fig = go.Figure(
        data=go.Bar(
            x=[row["presence"] for row in aspect_rows],
            y=[row["aspect"] for row in aspect_rows],
            orientation="h",
            marker=dict(color="#38bdf8"),
            text=[fmt_pct(row["presence"]) for row in aspect_rows],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        height=145,
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


def make_aspect_sentiment_figure(aspect_rows: list[dict]) -> go.Figure | None:
    mentioned_rows = [row for row in aspect_rows if row["mentioned"]]
    if not mentioned_rows:
        return None

    fig = go.Figure()
    fig.add_bar(
        x=[row["aspect"] for row in mentioned_rows],
        y=[row["neg"] for row in mentioned_rows],
        name=SENTIMENT_META[0]["label"],
        marker_color=SENTIMENT_META[0]["color"],
    )
    fig.add_bar(
        x=[row["aspect"] for row in mentioned_rows],
        y=[row["neu"] for row in mentioned_rows],
        name=SENTIMENT_META[1]["label"],
        marker_color=SENTIMENT_META[1]["color"],
    )
    fig.add_bar(
        x=[row["aspect"] for row in mentioned_rows],
        y=[row["pos"] for row in mentioned_rows],
        name=SENTIMENT_META[2]["label"],
        marker_color=SENTIMENT_META[2]["color"],
    )
    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        height=165,
        yaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    return fig


def make_aspect_dataframe(aspect_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Khia canh": row["aspect"],
                "De cap": "Co" if row["mentioned"] else "Khong",
                "Tin cay xuat hien": row["presence"],
                "Muc do": row["level_label"],
                "Tin cay muc do": row["level_conf"],
                "Tieu cuc": row["neg"],
                "Trung lap": row["neu"],
                "Tich cuc": row["pos"],
            }
            for row in aspect_rows
        ]
    )


def make_placeholder_aspect_rows() -> list[dict]:
    return [
        {
            "aspect": name,
            "presence": 0.0,
            "mentioned": False,
            "level_label": "Khong ap dung",
            "level_conf": 0.0,
            "neg": 0.0,
            "neu": 0.0,
            "pos": 0.0,
        }
        for name in ASPECT_NAMES
    ]


if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_updated" not in st.session_state:
    st.session_state.last_updated = ""
if "review_input" not in st.session_state:
    st.session_state.review_input = ""
if "model_label" not in st.session_state:
    st.session_state.model_label = MODEL_KEY_TO_LABEL["phobert"]
if "last_model_variant" not in st.session_state:
    st.session_state.last_model_variant = ""


def reset_dashboard_state():
    st.session_state.last_result = None
    st.session_state.last_updated = ""
    st.session_state.review_input = ""
    st.session_state.last_model_variant = ""


with st.container(border=True):
    st.markdown(
        """
        <div class="dashboard-header">
            <div class="dashboard-kicker">ABSA Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="panel-title">Input</div>', unsafe_allow_html=True)
        st.selectbox(
            "Model",
            options=list(MODEL_LABEL_TO_KEY.keys()),
            key="model_label",
            label_visibility="collapsed",
        )
        st.text_area(
            "Noi dung review",
            key="review_input",
            placeholder="Sach hay, giao hang nhanh, dong goi can than...",
            height=56,
            label_visibility="collapsed",
        )
        action_col_1, action_col_2 = st.columns(2, gap="small")
        with action_col_1:
            analyze = st.button("Phan tich", use_container_width=True, type="primary")
        with action_col_2:
            st.button("Xoa ket qua", use_container_width=True, on_click=reset_dashboard_state)

    if analyze:
        review = st.session_state.review_input.strip()
        if not review:
            st.warning("Vui long nhap noi dung review.")
        else:
            try:
                with st.spinner("Dang phan tich..."):
                    model_variant = MODEL_LABEL_TO_KEY[st.session_state.model_label]
                    predictor = load_predictor(model_variant)
                    st.session_state.last_result = predictor.predict(review)[0]
                st.session_state.last_model_variant = model_variant
                st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                st.error(f"Loi: {e}")

    active_model_variant = MODEL_LABEL_TO_KEY[st.session_state.model_label]
    if st.session_state.last_model_variant == active_model_variant:
        result = st.session_state.last_result
    else:
        result = None

    has_result = result is not None
    if has_result:
        aspect_rows = build_aspect_rows(result)
        overall = result["overall"]
        overall_meta = SENTIMENT_META[overall]
        overall_conf = max(result["overall_probs"])
        overall_probs = result["overall_probs"]
        present_count = sum(1 for row in aspect_rows if row["mentioned"])
        strongest_row = aspect_rows[0]
    else:
        aspect_rows = make_placeholder_aspect_rows()
        overall_meta = SENTIMENT_META[1]
        overall_conf = 0.0
        overall_probs = [0.0, 0.0, 0.0]
        present_count = 0
        strongest_row = {"aspect": "-", "presence": 0.0}

    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.markdown(stat_card("Tong the", overall_meta["label"], f"Do tin cay {fmt_pct(overall_conf)}"), unsafe_allow_html=True)
    k2.markdown(stat_card("Khia canh de cap", f"{present_count}/6", "So khia canh co nhac den"), unsafe_allow_html=True)
    k3.markdown(
        stat_card("Khia canh noi bat", strongest_row["aspect"], f"Tin cay xuat hien {fmt_pct(strongest_row['presence'])}"),
        unsafe_allow_html=True,
    )
    k4.markdown(
        stat_card("Model", st.session_state.model_label, st.session_state.last_updated or "-"),
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Tong quan", "Chi tiet khia canh"])

    with tabs[0]:
        chart_col_left, chart_col_right = st.columns(2, gap="medium")
        with chart_col_left:
            st.markdown('<div class="panel-title">Xac suat cam xuc tong the</div>', unsafe_allow_html=True)
            st.plotly_chart(make_overall_prob_figure(overall_probs), use_container_width=True)
        with chart_col_right:
            st.markdown('<div class="panel-title">Tin cay xuat hien theo khia canh</div>', unsafe_allow_html=True)
            st.plotly_chart(make_aspect_presence_figure(aspect_rows), use_container_width=True)

    with tabs[1]:
        st.markdown('<div class="panel-title">Muc do cam xuc theo tung khia canh</div>', unsafe_allow_html=True)
        sentiment_fig = make_aspect_sentiment_figure(aspect_rows)
        if sentiment_fig is not None:
            st.plotly_chart(sentiment_fig, use_container_width=True)
        else:
            st.caption("Dang cho du lieu phan tich.")

        st.dataframe(
            make_aspect_dataframe(aspect_rows),
            use_container_width=True,
            hide_index=True,
            height=125,
            column_config={
                "Khia canh": st.column_config.TextColumn(width="medium"),
                "De cap": st.column_config.TextColumn(width="small"),
                "Tin cay xuat hien": st.column_config.ProgressColumn(format="%.0f%%", min_value=0.0, max_value=1.0),
                "Muc do": st.column_config.TextColumn(width="small"),
                "Tin cay muc do": st.column_config.ProgressColumn(format="%.0f%%", min_value=0.0, max_value=1.0),
                "Tieu cuc": st.column_config.NumberColumn(format="%.0f%%", width="small"),
                "Trung lap": st.column_config.NumberColumn(format="%.0f%%", width="small"),
                "Tich cuc": st.column_config.NumberColumn(format="%.0f%%", width="small"),
            },
        )
