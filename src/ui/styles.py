CSS = """
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        height: 100vh;
        overflow: hidden !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0.45rem;
    }

    .block-container {
        box-sizing: border-box;
        padding-top: calc(1.55rem + 1cm);
        padding-bottom: 0.2rem;
        max-width: 70rem;
        padding-left: 0.55rem;
        padding-right: 0.55rem;
        margin-left: auto;
        margin-right: auto;
        height: calc(100vh - 0.45rem);
        overflow: hidden;
    }

    .dashboard-shell {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        padding: 1.6rem 1.05rem 1.05rem;
        background:
            linear-gradient(180deg, rgba(12, 18, 34, 0.98), rgba(6, 10, 18, 0.98)),
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.08), transparent 32%);
        box-shadow:
            0 22px 50px rgba(0, 0, 0, 0.32),
            0 1px 0 rgba(255, 255, 255, 0.06) inset;
        overflow: visible;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1.5px solid rgba(148, 163, 184, 0.46) !important;
        border-radius: 14px !important;
        background:
            linear-gradient(180deg, rgba(10, 16, 30, 0.94), rgba(7, 12, 24, 0.94)) !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.06),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2),
            0 14px 36px rgba(0, 0, 0, 0.18);
    }

    .dashboard-header {
        display: flex;
        flex-direction: column;
        gap: 0.08rem;
        padding: 0.3rem 0.1rem 0.22rem;
        margin-bottom: 0.28rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }

    .dashboard-kicker {
        display: block;
        font-size: 0.74rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: rgba(125, 211, 252, 0.88);
    }

    .dashboard-title {
        font-size: 1.12rem;
        line-height: 1.12;
        font-weight: 800;
        margin: 0.1rem 0 0;
    }

    .dashboard-subtitle {
        color: rgba(226, 232, 240, 0.72);
        font-size: 0.78rem;
    }

    .panel-card {
        height: 100%;
        padding: 0.85rem 0.9rem 0.9rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background:
            linear-gradient(180deg, rgba(16, 22, 38, 0.96), rgba(9, 14, 26, 0.96));
        box-shadow:
            0 10px 24px rgba(0, 0, 0, 0.24),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .panel-title {
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 0.32rem;
        color: rgba(248, 250, 252, 0.96);
    }

    .field-note {
        color: rgba(226, 232, 240, 0.7);
        font-size: 0.84rem;
        line-height: 1.3;
        margin-top: 0.28rem;
    }

    .stat-card {
        height: 100%;
        padding: 0.35rem 0.42rem;
        border-radius: 10px;
        border: 1px solid rgba(96, 165, 250, 0.18);
        background:
            linear-gradient(180deg, rgba(14, 22, 40, 0.98), rgba(8, 14, 26, 0.98));
        box-shadow:
            0 8px 18px rgba(0, 0, 0, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .stat-label {
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(148, 163, 184, 0.9);
        margin-bottom: 0.12rem;
    }

    .stat-value {
        font-size: 1.02rem;
        line-height: 1.05;
        font-weight: 800;
        color: #f8fafc;
    }

    .stat-hint {
        margin-top: 0.1rem;
        font-size: 0.68rem;
        color: rgba(74, 222, 128, 0.95);
    }

    .summary-line {
        margin: 0 0 0.18rem 0;
        color: rgba(226, 232, 240, 0.9);
        font-size: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .stTextArea textarea {
        min-height: 56px !important;
        max-height: 56px !important;
        line-height: 1.3 !important;
        font-size: 0.88rem !important;
        border-radius: 11px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .stButton > button {
        border-radius: 9px;
        padding-top: 0.28rem;
        padding-bottom: 0.28rem;
        font-weight: 700;
        font-size: 0.88rem;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
    }

    .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 0.2rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
        font-size: 0.9rem;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.1rem;
    }

    div[data-testid="stDataFrame"] {
        font-size: 0.82rem;
    }
</style>
"""
