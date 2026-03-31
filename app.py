import html

import plotly.graph_objects as go
import streamlit as st

from src.models.predictor import ABSAPredictor, ASPECT_COLS, ASPECT_NAMES
from src.ui.styles import CSS


st.set_page_config(page_title="ABSA Demo", layout="wide")
if CSS:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return ABSAPredictor()


SENTIMENT_META = {
    0: {"label": "Tiêu cực", "color": "#f87171"},
    1: {"label": "Trung lập", "color": "#fbbf24"},
    2: {"label": "Tích cực", "color": "#4ade80"},
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


with st.container(border=True):
    st.markdown(
        """
        <div class="dashboard-header">
            <div class="dashboard-kicker">ABSA Report</div>
            <div class="dashboard-title">Phân tích Cảm xúc Đánh giá Sách Tiki</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.14, 1.36], gap="large", vertical_alignment="top")

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Nhập đánh giá</div>', unsafe_allow_html=True)
            review = st.text_area("Nội dung", placeholder="Sách hay, giao hàng nhanh...", height=82, label_visibility="collapsed")
            analyze = st.button("Phân tích", use_container_width=True)

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Kết quả phân tích</div>', unsafe_allow_html=True)

            if analyze and review.strip():
                try:
                    predictor = load_predictor()
                    result = predictor.predict(review)[0]

                    overall = result["overall"]
                    overall_meta = SENTIMENT_META[overall]
                    overall_conf = max(result["overall_probs"])
                    present_count = sum(1 for col_key in ASPECT_COLS if result["aspects"][col_key] != -1)

                    strongest_idx = max(
                        range(len(ASPECT_COLS)),
                        key=lambda i: result["aspect_probs"][ASPECT_COLS[i]]["presence"],
                    )
                    strongest_aspect = ASPECT_NAMES[strongest_idx]
                    strongest_conf = result["aspect_probs"][ASPECT_COLS[strongest_idx]]["presence"]

                    c1, c2, c3 = st.columns(3, gap="small")
                    c1.markdown(stat_card("Tổng thể", overall_meta["label"], f"Độ tin cậy {fmt_pct(overall_conf)}"), unsafe_allow_html=True)
                    c2.markdown(stat_card("Khía cạnh", f"{present_count}/6", "Được nhận diện"), unsafe_allow_html=True)
                    c3.markdown(stat_card("Nổi bật", strongest_aspect, f"Độ tin cậy {fmt_pct(strongest_conf)}"), unsafe_allow_html=True)

                    st.markdown(
                        f'<div class="summary-line"><strong>Tổng thể:</strong> {html.escape(overall_meta["label"])} '
                        f'| <strong>Độ tin cậy:</strong> {fmt_pct(overall_conf)} '
                        f'| <strong>Xác suất:</strong> Tiêu cực {fmt_pct(result["overall_probs"][0])}, '
                        f'Trung lập {fmt_pct(result["overall_probs"][1])}, Tích cực {fmt_pct(result["overall_probs"][2])}</div>',
                        unsafe_allow_html=True,
                    )

                    probs = result["aspect_probs"]
                    aspect_data = []
                    for i, col_key in enumerate(ASPECT_COLS):
                        aspect_data.append(
                            {
                                "name": ASPECT_NAMES[i],
                                "conf": probs[col_key]["presence"],
                                "status": "Không nhắc" if result["aspects"][col_key] == -1 else SENTIMENT_META[result["aspects"][col_key]]["label"],
                            }
                        )
                    aspect_data = sorted(aspect_data, key=lambda x: x["conf"], reverse=True)

                    chart_col, bar_col = st.columns([1.05, 1.0], gap="medium")
                    with chart_col:
                        radar_values = [probs[c]["presence"] for c in ASPECT_COLS] + [probs[ASPECT_COLS[0]]["presence"]]
                        radar_labels = list(ASPECT_NAMES) + [ASPECT_NAMES[0]]
                        radar_fig = go.Figure(
                            data=go.Scatterpolar(
                                r=radar_values,
                                theta=radar_labels,
                                fill="toself",
                                fillcolor="rgba(56,189,248,0.18)",
                                line=dict(color="#38bdf8", width=2.2),
                            )
                        )
                        radar_fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            polar=dict(
                                bgcolor="rgba(11, 17, 32, 0.95)",
                                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
                                angularaxis=dict(tickfont=dict(size=10)),
                            ),
                            margin=dict(l=24, r=12, t=6, b=6),
                            height=165,
                            showlegend=False,
                        )
                        st.plotly_chart(radar_fig, use_container_width=True)

                    with bar_col:
                        bar_fig = go.Figure(
                            data=go.Bar(
                                x=[row["conf"] for row in aspect_data],
                                y=[row["name"] for row in aspect_data],
                                orientation="h",
                                marker=dict(color="#38bdf8"),
                                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                            )
                        )
                        bar_fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=8, r=8, t=6, b=6),
                            height=165,
                            xaxis=dict(range=[0, 1], tickformat=".0%"),
                            yaxis=dict(autorange="reversed"),
                            showlegend=False,
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)

                    top_summary = " · ".join(f"{row['name']} {fmt_pct(row['conf'])}" for row in aspect_data[:3])
                    st.markdown(
                        f'<div class="summary-line"><strong>Khía cạnh nổi bật:</strong> {html.escape(top_summary)}</div>',
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"Lỗi: {e}")
            elif analyze:
                st.warning("Vui lòng nhập nội dung.")
            else:
                st.info("Nhập đánh giá bên trái rồi bấm Phân tích.")
