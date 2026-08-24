"""Migrated from 02_eda_detailed_visualization.ipynb.

This file preserves the original experiment code for audit/reproducibility.
The production benchmark uses the modular ml/ package instead.
"""


# %% [code cell 1]
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from pathlib import Path
from functools import lru_cache
import math
import re
from collections import Counter
import warnings
import unicodedata

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import PercentFormatter
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

from pyvi import ViTokenizer
from wordcloud import WordCloud

sns.set_theme(style="whitegrid", context="notebook")

def find_font_family():
    for path in [
        Path(r"C:/Windows/Fonts/arial.ttf"),
        Path(r"C:/Windows/Fonts/tahoma.ttf"),
        Path(r"C:/Windows/Fonts/segoeui.ttf"),
    ]:
        if path.exists():
            return fm.FontProperties(fname=str(path)).get_name(), str(path)
    return "DejaVu Sans", None

FONT_FAMILY, WORDCLOUD_FONT = find_font_family()
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": FONT_FAMILY,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
    }
)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 80)

SPLITS = ["train", "val", "test"]
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_LABELS = {"negative": "Tiêu cực", "neutral": "Trung lập", "positive": "Tích cực"}
SENTIMENT_COLORS = {"negative": "#d64b4b", "neutral": "#f0ad4e", "positive": "#2ca02c"}
ASPECT_COLS = [
    "as_content",
    "as_physical",
    "as_price",
    "as_packaging",
    "as_delivery",
    "as_service",
]
ASPECT_LABELS = {
    "as_content": "Nội dung",
    "as_physical": "Chất lượng",
    "as_price": "Giá",
    "as_packaging": "Đóng gói",
    "as_delivery": "Giao hàng",
    "as_service": "Dịch vụ",
}
ASPECT_COLORS = {
    "as_content": "#2d6cdf",
    "as_physical": "#7a4cc2",
    "as_price": "#f39c12",
    "as_packaging": "#d64b4b",
    "as_delivery": "#2ca02c",
    "as_service": "#17a2b8",
}
RANDOM_STATE = 42

#Hàm tự tìm thư mục gốc
def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd.parent, cwd.parent.parent]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    return cwd

ROOT = find_project_root()

#Hàm chuẫn hóa văn bản
def normalize_text(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return " ".join(text.split())
 
@lru_cache(maxsize=20000) # Lưu kết quả tách từ vào cache để xử lý cực nhanh các câu trùng lặp
def segment_text(text: str) -> str:
    text = normalize_text(text).lower()
    segmented = ViTokenizer.tokenize(text)
    tokens = re.findall(r"[\w_]+", segmented, flags=re.UNICODE)
    tokens = [tok for tok in tokens if not tok.isdigit() and len(tok) > 1]
    return " ".join(tokens)

def tokenize_for_analysis(text: str) -> str:
    return segment_text(text)

def tokenize_list(text: str) -> list[str]:
    return tokenize_for_analysis(text).split()

def show(fig):
    display(fig)
    plt.close(fig)
# Hàm tự động ghi con số cụ thể lên phía trên các cột biểu đồ
def annotate_bars(ax, bars, values, fmt="{:,.0f}"):
    values = list(values)
    peak = max(values) if values else 0
    offset = peak * 0.015 if peak else 0.01
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

def plot_lollipop(ax, labels, values, title, xlabel, color="#2d6cdf", fmt="{:,.0f}"):
    x = np.arange(len(values))
    markerline, stemlines, baseline = ax.stem(x, values, basefmt=" ")
    plt.setp(markerline, markersize=8, markerfacecolor=color, markeredgecolor=color)
    plt.setp(stemlines, color=color, linewidth=2.2)
    plt.setp(baseline, visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="y", color="#dfe3eb", linewidth=0.8, alpha=0.8)
    sns.despine(ax=ax)
    peak = max(values) if len(values) else 0
    offset = peak * 0.015 if peak else 0.01
    for xi, yi in zip(x, values):
        ax.text(xi, yi + offset, fmt.format(yi), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, peak * 1.18 if peak else 1)

def plot_heatmap(ax, data, title, xlabel, ylabel, fmt=".2f", cmap="YlGnBu", vmin=None, vmax=None):
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": ""},
        ax=ax,
    )
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def make_wordcloud(text: str) -> WordCloud:
    return WordCloud(
        width=900,
        height=500,
        background_color="white",
        collocations=False,
        max_words=120,
        prefer_horizontal=0.9,
        random_state=RANDOM_STATE,
        font_path=WORDCLOUD_FONT,
        colormap="viridis",
    ).generate(text if text.strip() else "empty")

def top_terms(texts, ngram_range=(1, 1), top_n=30, min_df=5, binary=False):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
        binary=binary,
    )
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return pd.DataFrame(columns=["term", "count"])
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    frame = pd.DataFrame({"term": vectorizer.get_feature_names_out(), "count": counts})
    return frame.sort_values("count", ascending=False).head(top_n)

def dominant_aspect(row):
    vals = row[ASPECT_COLS].dropna()
    if vals.empty:
        return "Không nhãn"
    return ASPECT_LABELS[vals.astype(float).idxmax()]

def compute_conflict(row):
    vals = row[ASPECT_COLS].dropna()
    if len(vals) < 2:
        return False
    return len(set(vals.astype(int))) >= 2

def load_pmi_table(frame):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_selection import chi2

    token_series = frame["analysis_text"]
    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        min_df=10,
        binary=True,
    )
    X = vectorizer.fit_transform(token_series)
    y = frame["sentiment_name"].map({"negative": 0, "neutral": 1, "positive": 2}).to_numpy()
    chi2_score, chi2_p = chi2(X, y)
    terms = vectorizer.get_feature_names_out()
    class_counts = frame["sentiment_name"].value_counts().reindex(SENTIMENT_ORDER)
    p_class = class_counts / len(frame)
    term_df = np.asarray(X.sum(axis=0)).ravel() / X.shape[0]
    records = []
    for cls_idx, cls in enumerate(SENTIMENT_ORDER):
        mask = (frame["sentiment_name"] == cls).to_numpy()
        class_prob = p_class[cls]
        term_prob_given_class = np.asarray(X[mask].sum(axis=0)).ravel() / mask.sum()
        pmi = np.log2((term_prob_given_class * class_prob + 1e-12) / (term_df * class_prob + 1e-12))
        for term, pmi_val, chi_val, p_val, dfreq in zip(terms, pmi, chi2_score, chi2_p, np.asarray(X.sum(axis=0)).ravel()):
            records.append(
                {
                    "term": term,
                    "class": cls,
                    "pmi": float(pmi_val),
                    "chi2": float(chi_val),
                    "p_value": float(p_val),
                    "df": int(dfreq),
                }
            )
    return pd.DataFrame(records)


# %% [code cell 2]
raw_paths = {
    "train": ROOT / "data" / "processed" / "train_clean.json",
    "val": ROOT / "data" / "processed" / "val_clean.json",
    "test": ROOT / "data" / "processed" / "test_clean.json",
}

frames = []
for split, path in raw_paths.items():
    frame = pd.read_json(path)
    frame["split"] = split
    frames.append(frame)

df_all = pd.concat(frames, ignore_index=True)
df_all["sentiment_name"] = df_all["sentiment"].map(SENTIMENT_MAP).fillna("unknown")
df_all["content"] = df_all["content"].map(normalize_text)
df_all["seg_text"] = df_all["content"].map(segment_text)
df_all["analysis_text"] = df_all["seg_text"].map(tokenize_for_analysis)
df_all["analysis_tokens"] = df_all["analysis_text"].str.split()
df_all["char_count"] = df_all["content"].str.len()
df_all["word_count_raw"] = df_all["content"].str.split().str.len()
df_all["aspect_count"] = df_all[ASPECT_COLS].notna().sum(axis=1)
df_all["punct_density"] = df_all["content"].str.count(r"[^\w\s]") / df_all["char_count"].clip(lower=1)
df_all["dominant_aspect"] = df_all.apply(dominant_aspect, axis=1)
df_all["conflict"] = df_all.apply(compute_conflict, axis=1)

df_train = df_all[df_all["split"] == "train"].copy()
df_train["analysis_text"] = df_train["analysis_text"].astype(str)
df_all["analysis_text"] = df_all["analysis_text"].astype(str)

# Default EDA uses train; split-level statistics use df_all separately.
df_eda = df_train

split_overview = (
    df_all.groupby("split")
    .agg(
        Records=("review_id", "size"),
        Avg_Len=("char_count", "mean"),
        Median_Len=("char_count", "median"),
    )
    .reindex(SPLITS)
    .reset_index()
)
split_overview["Avg_Len"] = split_overview["Avg_Len"].round(1)
split_overview["Median_Len"] = split_overview["Median_Len"].round(0).astype(int)

sentiment_counts = df_eda["sentiment_name"].value_counts().reindex(SENTIMENT_ORDER)
sentiment_share = sentiment_counts / len(df_eda)
imbalance_ratio = sentiment_counts["negative"] / sentiment_counts["positive"]

aspect_coverage = pd.DataFrame(
    {
        "aspect": [ASPECT_LABELS[c] for c in ASPECT_COLS],
        "coverage_pct": [df_eda[c].notna().mean() * 100 for c in ASPECT_COLS],
    }
).sort_values("coverage_pct", ascending=True)

aspect_missing_rate = (
    df_all.groupby("split")[ASPECT_COLS]
    .apply(lambda x: x.isna().mean() * 100)
    .reindex(SPLITS)
)

aspect_count_dist = df_eda["aspect_count"].value_counts().sort_index()
multi_aspect_share = (df_eda["aspect_count"] >= 2).mean()
conflict_rate = df_eda["conflict"].mean()
split_sentiment_p = pd.crosstab(df_all["split"], df_all["sentiment_name"], normalize="index").reindex(SPLITS)
chi2_table = pd.crosstab(df_all["split"], df_all["sentiment_name"])

display(pd.DataFrame(
    {
        "total_reviews": [f"{len(df_all):,}"],
        "train": [f"{len(df_train):,}"],
        "val": [f"{len(df_all[df_all.split == 'val']):,}"],
        "test": [f"{len(df_all[df_all.split == 'test']):,}"],
        "negative_rate": [f"{sentiment_share['negative']:.1%}"],
        "neutral_rate": [f"{sentiment_share['neutral']:.1%}"],
        "positive_rate": [f"{sentiment_share['positive']:.1%}"],
    }
))


# %% [code cell 3]
display(split_overview)


# %% [code cell 4]
display_cols = ["sentiment", *ASPECT_COLS]
missing_rate = df_all.groupby("split")[display_cols].apply(lambda x: x.isna().mean() * 100).reindex(SPLITS)

fig, ax = plt.subplots(figsize=(10, 4.8))
sns.heatmap(
    missing_rate,
    annot=True,
    fmt=".1f",
    cmap="Reds",
    vmin=0,
    vmax=100,
    cbar_kws={"label": "Tỷ lệ thiếu (%)"},
    ax=ax,
)
ax.set_title("Tỷ lệ thiếu theo split × sentiment/aspect", loc="left")
ax.set_xlabel("Khía cạnh")
ax.set_ylabel("Split")
ax.set_xticklabels(["Cảm xúc chung", *[ASPECT_LABELS[col] for col in ASPECT_COLS]], rotation=20, ha="right")
show(fig)


# %% [code cell 5]
fig, ax = plt.subplots(figsize=(9, 5.2))
plot_lollipop(
    ax,
    aspect_coverage["aspect"].tolist(),
    aspect_coverage["coverage_pct"].to_numpy(),
    "Độ phủ nhãn theo aspect",
    "Khía cạnh",
    color="#2d6cdf",
    fmt="{:.1f}%",
)
ax.set_ylabel("Độ phủ (%)")
show(fig)


# %% [code cell 6]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))

# --- Bar chart ---
labels = [SENTIMENT_LABELS[s] for s in SENTIMENT_ORDER]
values = list(sentiment_counts.values)
colors = [SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]

bars = ax1.bar(labels, values, color=colors)
annotate_bars(ax1, bars, values)

ax1.set_title("Phân phối sentiment", loc="left")
ax1.set_xlabel("")
ax1.set_ylabel("Số review")
ax1.grid(axis="y", color="#dfe3eb", linewidth=0.8, alpha=0.8)

# --- Pie chart ---
ax2.pie(
    values,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False
)

ax2.set_title("Tỷ lệ sentiment (%)", loc="left")

plt.tight_layout()
show(fig)


# %% [code cell 7]
aspect_ratio = pd.DataFrame(
    {
        "aspect": [ASPECT_LABELS[c] for c in ASPECT_COLS],
        "negative": [df_eda[c].eq(0).mean() * 100 for c in ASPECT_COLS],
        "neutral": [df_eda[c].eq(1).mean() * 100 for c in ASPECT_COLS],
        "positive": [df_eda[c].eq(2).mean() * 100 for c in ASPECT_COLS],
    }
).set_index("aspect")

fig, ax = plt.subplots(figsize=(9.5, 4.8))
sns.heatmap(
    aspect_ratio.T,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    vmin=0,
    vmax=100,
    cbar_kws={"label": "Rate (%)"},
    ax=ax,
)
ax.set_title("Sentiment by aspect", loc="left")
ax.set_xlabel("Aspect")
ax.set_ylabel("Sentiment")
ax.set_yticklabels([SENTIMENT_LABELS[s] for s in SENTIMENT_ORDER], rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
show(fig)


# %% [code cell 8]
fig, ax = plt.subplots(figsize=(10, 5.2))
bottom = np.zeros(len(aspect_ratio))
for sentiment in SENTIMENT_ORDER:
    ax.bar(
        aspect_ratio.index,
        aspect_ratio[sentiment],
        bottom=bottom,
        color=SENTIMENT_COLORS[sentiment],
        label=SENTIMENT_LABELS[sentiment],
    )
    bottom += aspect_ratio[sentiment].to_numpy()
ax.set_title("Cơ cấu sentiment trong từng aspect", loc="left")
ax.set_ylabel("Tỷ lệ (%)")
ax.tick_params(axis="x", rotation=20)
ax.legend(ncol=3, loc="upper right")
ax.grid(axis="y", color="#dfe3eb", linewidth=0.8, alpha=0.8)
ax.yaxis.set_major_formatter(PercentFormatter(100))
sns.despine(ax=ax)
show(fig)


# %% [code cell 9]
aspect_share_by_count = pd.crosstab(df_eda["aspect_count"], df_eda["sentiment_name"], normalize="index").reindex(columns=SENTIMENT_ORDER).fillna(0)
cooc = df_eda[ASPECT_COLS].notna().astype(int).T @ df_eda[ASPECT_COLS].notna().astype(int)
cooc_pct = cooc / len(df_eda) * 100
conflict_by_count = df_eda.groupby("aspect_count")["conflict"].mean().reindex(aspect_count_dist.index) * 100

fig, ax = plt.subplots(figsize=(8.5, 4.8))
bars = ax.bar(aspect_count_dist.index.astype(str), aspect_count_dist.values, color="#2d6cdf")
annotate_bars(ax, bars, aspect_count_dist.values)
ax.set_title("Distribution of aspects per review", loc="left")
ax.set_xlabel("Number of aspects")
ax.set_ylabel("Review count")
ax.grid(axis="y", color="#dfe3eb", linewidth=0.8, alpha=0.8)
show(fig)


# %% [code cell 10]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
bars = ax.bar(conflict_by_count.index.astype(str), conflict_by_count.values, color="#d64b4b")
annotate_bars(ax, bars, conflict_by_count.values, fmt="{:.1f}%")
ax.set_title("Tỷ lệ xung đột giữa các aspect", loc="left")
ax.set_xlabel("Số aspect")
ax.set_ylabel("Tỷ lệ conflict (%)")
ax.grid(axis="y", color="#dfe3eb", linewidth=0.8, alpha=0.8)
show(fig)


# %% [code cell 11]
fig, ax = plt.subplots(figsize=(8.8, 4.8))
sns.histplot(df_eda["char_count"], bins=40, kde=True, color="#2d6cdf", ax=ax)
ax.set_title("Histogram + KDE of review length", loc="left")
ax.set_xlabel("Characters")
ax.set_ylabel("Review count")
show(fig)


# %% [code cell 12]
fig, ax = plt.subplots(figsize=(8.8, 4.8))
sns.violinplot(
    data=df_eda,
    x="sentiment_name",
    y="char_count",
    order=SENTIMENT_ORDER,
    hue="sentiment_name",
    hue_order=SENTIMENT_ORDER,
    palette=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
    inner="quartile",
    cut=0,
    dodge=False,
    ax=ax,
)
leg = ax.get_legend()
if leg is not None:
    leg.remove()
ax.set_xticks(range(len(SENTIMENT_ORDER)))
ax.set_xticklabels([SENTIMENT_LABELS[s] for s in SENTIMENT_ORDER])
ax.set_title("Length by sentiment", loc="left")
ax.set_xlabel("")
ax.set_ylabel("Characters")
show(fig)


# %% [code cell 13]
top_word_frame = top_terms(df_train["analysis_text"], ngram_range=(1, 1), top_n=30, min_df=10)
fig, ax = plt.subplots(figsize=(9, 7))
data = top_word_frame.sort_values("count", ascending=True)
bars = ax.barh(data["term"], data["count"], color="#2d6cdf")
ax.set_title("Top 30 từ nổi bật trong train", loc="left")
ax.set_xlabel("Số lần xuất hiện")
ax.set_ylabel("")
ax.grid(axis="x", color="#dfe3eb", linewidth=0.8, alpha=0.8)
sns.despine(ax=ax)
for bar, value in zip(bars, data["count"]):
    ax.text(bar.get_width() + data["count"].max() * 0.01, bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", fontsize=9)
show(fig)


# %% [code cell 14]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
for ax, sentiment in zip(axes, SENTIMENT_ORDER):
    text = " ".join(df_train.loc[df_train["sentiment_name"] == sentiment, "analysis_text"])
    cloud = make_wordcloud(text)
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_title(f"{SENTIMENT_LABELS[sentiment]} - Word cloud", loc="left")
    ax.axis("off")
show(fig)


# %% [code cell 15]
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
axes = axes.flatten()
for ax, col in zip(axes, ASPECT_COLS):
    text = " ".join(df_train.loc[df_train[col].notna(), "analysis_text"])
    cloud = make_wordcloud(text)
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_title(ASPECT_LABELS[col], loc="left")
    ax.axis("off")
show(fig)


# %% [code cell 16]
from matplotlib.patches import Patch

sentiment_specs = [
    ("negative", "Tiêu cực"),
    ("neutral", "Trung lập"),
    ("positive", "Tích cực"),
]

fig, axes = plt.subplots(3, 1, figsize=(10, 13), constrained_layout=True)

for ax, (sentiment, sentiment_label) in zip(axes, sentiment_specs):
    texts = df_train.loc[df_train["sentiment_name"] == sentiment, "analysis_text"].tolist()
    frame = top_terms(texts, ngram_range=(2, 2), top_n=15, min_df=5)
    data = frame.sort_values("count", ascending=True)
    bars = ax.barh(data["term"], data["count"], color=SENTIMENT_COLORS[sentiment])
    ax.set_title(f"{sentiment_label} - Bigram", loc="left")
    ax.set_xlabel("Số lần xuất hiện")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#dfe3eb", linewidth=0.8, alpha=0.8)
    sns.despine(ax=ax)
    if len(data):
        peak = float(data["count"].max())
        for bar, value in zip(bars, data["count"]):
            ax.text(
                bar.get_width() + peak * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value)}",
                va="center",
                fontsize=8,
            )

show(fig)


# %% [code cell 17]
from matplotlib.patches import Patch

sentiment_specs = [
    ("negative", "Tiêu cực"),
    ("neutral", "Trung lập"),
    ("positive", "Tích cực"),
]

fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

for ax, (sentiment, sentiment_label) in zip(axes, sentiment_specs):
    texts = df_train.loc[df_train["sentiment_name"] == sentiment, "analysis_text"].tolist()
    frame = top_terms(texts, ngram_range=(3, 3), top_n=8, min_df=5)
    data = frame.sort_values("count", ascending=True)
    bars = ax.barh(data["term"], data["count"], color=SENTIMENT_COLORS[sentiment])
    ax.set_title(f"{sentiment_label} - Trigram", loc="left")
    ax.set_xlabel("Số lần xuất hiện")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#dfe3eb", linewidth=0.8, alpha=0.8)
    sns.despine(ax=ax)
    if len(data):
        peak = float(data["count"].max())
        for bar, value in zip(bars, data["count"]):
            ax.text(
                bar.get_width() + peak * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value)}",
                va="center",
                fontsize=8,
            )

show(fig)


# %% [code cell 18]
aspect_corr = df_eda[ASPECT_COLS].corr(method="spearman").rename(index=ASPECT_LABELS, columns=ASPECT_LABELS)
fig, ax = plt.subplots(figsize=(7.8, 5.6))
sns.heatmap(
    aspect_corr,
    annot=True,
    fmt=".2f",
    cmap="PuBuGn",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Spearman"},
    ax=ax,
)
ax.set_title("Spearman correlation between aspects", loc="left")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
show(fig)


# %% [code cell 19]
conflict_by_sentiment = df_eda.groupby("sentiment_name")["conflict"].mean().reindex(SENTIMENT_ORDER) * 100
fig, ax = plt.subplots(figsize=(7.8, 5.2))
bars = ax.bar(
    [SENTIMENT_LABELS[s] for s in SENTIMENT_ORDER],
    conflict_by_sentiment.values,
    color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
)
annotate_bars(ax, bars, conflict_by_sentiment.values, fmt="{:.1f}%")
ax.set_title("Conflict rate by sentiment", loc="left")
ax.set_ylabel("Conflict rate (%)")
ax.grid(axis="y", color="#dfe3eb", linewidth=0.8, alpha=0.8)
show(fig)


# %% [code cell 20]
from scipy.stats import chi2_contingency

fig, ax = plt.subplots(figsize=(8.8, 5.2))
for split, color in zip(SPLITS, ["#2d6cdf", "#f0ad4e", "#2ca02c"]):
    subset = df_all[df_all["split"] == split]
    sns.kdeplot(subset["char_count"], ax=ax, label=split, color=color, linewidth=2.2)
ax.set_title("Overlay phân phối độ dài review theo split", loc="left")
ax.set_xlabel("Số ký tự")
ax.set_ylabel("Mật độ")
ax.legend(title="Split")
show(fig)


# %% [code cell 21]
chi2, p_value, dof, _ = chi2_contingency(chi2_table)
overlap_pairs = []
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    a_ids = set(df_all.loc[df_all["split"] == a, "review_id"])
    b_ids = set(df_all.loc[df_all["split"] == b, "review_id"])
    overlap_pairs.append({"Cặp split": f"{a} × {b}", "Overlap review_id": len(a_ids & b_ids)})
overlap_df = pd.DataFrame(overlap_pairs)

display(pd.DataFrame(
    {
        "Chi2": [round(chi2, 4)],
        "p-value": [p_value],
        "dof": [dof],
        "Sentiment ổn định?": ["Có" if p_value > 0.05 else "Không"],
    }
))
display(overlap_df)


# %% [code cell 22]
from sklearn.feature_extraction.text import CountVectorizer

pmi_df = load_pmi_table(df_train)
term_summary = (
    pmi_df.groupby("term")
    .agg(
        max_pmi=("pmi", "max"),
        min_pmi=("pmi", "min"),
        max_abs_pmi=("pmi", lambda s: s.abs().max()),
        chi2=("chi2", "max"),
        p_value=("p_value", "min"),
        df=("df", "max"),
    )
    .reset_index()
)
term_summary["pmi_gap"] = term_summary["max_pmi"] - term_summary["min_pmi"]
pivot = pmi_df.pivot(index="term", columns="class", values="pmi").fillna(0)
pivot["delta_pos_neg"] = pivot["positive"] - pivot["negative"]
diverging = pivot.sort_values("delta_pos_neg")
diverging = pd.concat([diverging.head(15), diverging.tail(15)])


# %% [code cell 23]
fig, ax = plt.subplots(figsize=(10.5, 7))
data = diverging["delta_pos_neg"].sort_values()
colors = ["#d64b4b" if v < 0 else "#2ca02c" for v in data.values]
bars = ax.barh(data.index, data.values, color=colors)
ax.axvline(0, color="#333333", linewidth=1)
ax.set_title("PMI khác biệt Positive vs Negative", loc="left")
ax.set_xlabel("PMI Positive - PMI Negative")
ax.set_ylabel("")
ax.grid(axis="x", color="#dfe3eb", linewidth=0.8, alpha=0.8)
sns.despine(ax=ax)
show(fig)


# %% [code cell 24]
fig, ax = plt.subplots(figsize=(10.5, 7))
heat_terms = term_summary.sort_values("max_abs_pmi", ascending=False).head(50)["term"]
sns.heatmap(
    pivot.loc[heat_terms, SENTIMENT_ORDER],
    cmap="RdYlGn",
    center=0,
    ax=ax,
    cbar_kws={"label": "PMI"},
)
ax.set_title("Ma trận PMI của 50 term nổi bật nhất", loc="left")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Term")
ax.set_xticklabels([SENTIMENT_LABELS[s] for s in SENTIMENT_ORDER], rotation=0)
show(fig)


# %% [code cell 25]
fig, ax = plt.subplots(figsize=(10.5, 6.5))
bubble = term_summary.sort_values("max_abs_pmi", ascending=False).head(80).copy()
ax.scatter(
    bubble["df"],
    bubble["max_abs_pmi"],
    s=np.clip(-np.log10(bubble["p_value"] + 1e-12) * 50, 20, 800),
    c=bubble["max_pmi"],
    cmap="coolwarm",
    alpha=0.75,
    edgecolors="white",
    linewidths=0.5,
)
ax.set_title("Frequency - PMI - chi-square p-value", loc="left")
ax.set_xlabel("Document frequency")
ax.set_ylabel("Max |PMI|")
ax.grid(axis="both", color="#dfe3eb", linewidth=0.8, alpha=0.8)
show(fig)


# %% [code cell 26]
import networkx as nx
import matplotlib.patches as mpatches

# 1. Chuẩn bị dữ liệu (Giữ nguyên logic của bạn nhưng tối ưu hóa)
token_counter = Counter()
sentiment_token_counter = {sentiment: Counter() for sentiment in SENTIMENT_ORDER}
for sentiment, tokens in zip(df_train["sentiment_name"], df_train["analysis_tokens"]):
    token_counter.update(tokens)
    sentiment_token_counter[sentiment].update(tokens)

# Lấy top 30 thay vì 25 để đồ thị "dày" và chuyên nghiệp hơn
top_tokens = [token for token, _ in token_counter.most_common(30)]
cooc_counter = Counter()
for tokens in df_train["analysis_tokens"]:
    present = sorted(set(token for token in tokens if token in top_tokens))
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            cooc_counter[(present[i], present[j])] += 1

# 2. Xây dựng đồ thị
G = nx.Graph()
for token in top_tokens:
    # Xác định sentiment chiếm ưu thế cho từ này
    dominant = max(SENTIMENT_ORDER, key=lambda s: sentiment_token_counter[s][token])
    G.add_node(token, freq=token_counter[token], sentiment=dominant)

# Chỉ lấy các cạnh có trọng số đủ lớn để tránh rối đồ thị
threshold = 28 
for (a, b), weight in cooc_counter.items():
    if weight >= threshold:
        G.add_edge(a, b, weight=weight)

# 3. Trực quan hóa "Premium Look"
fig, ax = plt.subplots(figsize=(14, 10), facecolor='#FAFAFA') # Nền xám cực nhẹ
pos = nx.spring_layout(G, seed=RANDOM_STATE, k=0.85, iterations=50)

# Tính toán các thuộc tính trực quan
node_colors = [SENTIMENT_COLORS[G.nodes[n]["sentiment"]] for n in G.nodes]
# Dùng sqrt để scale kích thước node mượt hơn
node_sizes = [np.sqrt(G.nodes[n]["freq"]) * 140 for n in G.nodes]
# Scale độ dày cạnh
edge_widths = [np.log1p(G[u][v]["weight"] / threshold) * 2.5 for u, v in G.edges]

# Vẽ các thành phần
# Vẽ cạnh trước (alpha thấp để làm nền)
nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.15, 
                       edge_color="#455A64", style='solid')

# Vẽ node có viền trắng tinh tế
nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                               node_size=node_sizes, alpha=0.95, 
                               linewidths=1.5, edgecolors='#FFFFFF')

# Vẽ nhãn (labels) với font đẹp và shadow nhẹ (nếu cần)
labels = nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, 
                                 font_weight='bold', font_family='sans-serif',
                                 font_color='#263238')

# 4. Thêm Legend (Cực kỳ quan trọng để bảo vệ đồ án)
legend_handles = [
    mpatches.Patch(color=SENTIMENT_COLORS["negative"], label="Tiêu cực"),
    mpatches.Patch(color=SENTIMENT_COLORS["neutral"], label="Trung tính"),
    mpatches.Patch(color=SENTIMENT_COLORS["positive"], label="Tích cực")
]
ax.legend(handles=legend_handles, loc='upper right', title="Cảm xúc chủ đạo", 
          frameon=True, facecolor='white', shadow=True, fontsize=10)

# Tinh chỉnh tiêu đề và layout
ax.set_title("Mối liên kết từ ngữ và Cảm xúc (Co-occurrence Network)", 
             fontsize=16, fontweight='bold', pad=20, color='#263238')
ax.text(0.01, -0.01, f"* Kích thước node: Tần suất từ | Độ dày cạnh: Tần suất xuất hiện cùng nhau (Threshold > {threshold})", 
        transform=ax.transAxes, fontsize=9, color='#78909C', style='italic')

plt.axis("off")
plt.tight_layout()
plt.show() # Hoặc show(fig) tùy hàm của bạn


# %% [code cell 27]
import plotly.graph_objects as go

aspect_flow = (
    df_train.melt(
        id_vars=["sentiment_name"],
        value_vars=ASPECT_COLS,
        var_name="aspect_col",
        value_name="aspect_value",
    )
    .dropna(subset=["aspect_value"])
    .groupby(["sentiment_name", "aspect_col"])
    .size()
    .reset_index(name="count")
)
aspect_flow["aspect_label"] = aspect_flow["aspect_col"].map(ASPECT_LABELS)

labels = list(SENTIMENT_LABELS.values()) + [ASPECT_LABELS[c] for c in ASPECT_COLS]
color_map = [SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER] + [ASPECT_COLORS[c] for c in ASPECT_COLS]
label_idx = {label: idx for idx, label in enumerate(labels)}
sources, targets, values, link_colors = [], [], [], []
for _, row in aspect_flow.iterrows():
    sources.append(label_idx[SENTIMENT_LABELS[row["sentiment_name"]]])
    targets.append(label_idx[row["aspect_label"]])
    values.append(int(row["count"]))
    link_colors.append(SENTIMENT_COLORS[row["sentiment_name"]])

sankey = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=18,
                thickness=18,
                line=dict(color="white", width=0.5),
                label=labels,
                color=color_map,
            ),
            link=dict(source=sources, target=targets, value=values, color=link_colors),
        )
    ]
)
sankey.update_layout(
    title_text="Dòng chảy từ sentiment sang aspect",
    font=dict(size=12, family=FONT_FAMILY),
    height=550,
)
display(sankey)


# %% [code cell 28]
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

sample_parts = []
for sentiment in SENTIMENT_ORDER:
    part = df_train[df_train["sentiment_name"] == sentiment]
    sample_parts.append(part.sample(n=min(250, len(part)), random_state=RANDOM_STATE))
embed_df = pd.concat(sample_parts, ignore_index=True)
embed_df["dominant_aspect_label"] = embed_df.apply(dominant_aspect, axis=1)

EMBED_MODE = "TF-IDF/SVD"
tfidf = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, min_df=5, max_features=5000)
X = tfidf.fit_transform(embed_df["analysis_text"])
svd = TruncatedSVD(n_components=200, random_state=RANDOM_STATE)
embeddings = svd.fit_transform(X)

pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=RANDOM_STATE)
emb_50 = pca.fit_transform(embeddings)
reducer = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=RANDOM_STATE)
emb_2d = reducer.fit_transform(emb_50)

embed_df["x"] = emb_2d[:, 0]
embed_df["y"] = emb_2d[:, 1]


# %% [code cell 29]
fig, ax = plt.subplots(figsize=(8.2, 6.4))
sns.scatterplot(
    data=embed_df,
    x="x",
    y="y",
    hue="sentiment_name",
    hue_order=SENTIMENT_ORDER,
    palette=SENTIMENT_COLORS,
    alpha=0.75,
    s=35,
    linewidth=0,
    ax=ax,
)
ax.set_title(f"Không gian embedding - màu theo sentiment ({EMBED_MODE})", loc="left")
ax.set_xlabel("Trục 1")
ax.set_ylabel("Trục 2")
show(fig)


# %% [code cell 30]
fig, ax = plt.subplots(figsize=(8.2, 6.4))
sns.scatterplot(
    data=embed_df,
    x="x",
    y="y",
    hue="dominant_aspect_label",
    alpha=0.75,
    s=35,
    linewidth=0,
    ax=ax,
)
ax.set_title("Không gian embedding - màu theo aspect chi phối", loc="left")
ax.set_xlabel("Trục 1")
ax.set_ylabel("Trục 2")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
show(fig)


# %% [code cell 31]
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pmi_pivot = pmi_df.pivot(index="term", columns="class", values="pmi").fillna(0)

def doc_pmi_score(tokens, sentiment_key):
    scores = [pmi_pivot.loc[token, sentiment_key] for token in tokens if token in pmi_pivot.index]
    return max(scores) if scores else 0.0

feature_df = pd.DataFrame(index=df_train.index)
feature_df["text_length"] = df_train["char_count"]
feature_df["word_count"] = df_train["analysis_tokens"].map(len)
feature_df["avg_word_length"] = feature_df["text_length"] / feature_df["word_count"].clip(lower=1)
feature_df["unique_word_ratio"] = df_train["analysis_tokens"].map(lambda tokens: len(set(tokens)) / len(tokens) if len(tokens) else 0)
feature_df["aspect_count"] = df_train["aspect_count"]
feature_df["punctuation_density"] = df_train["punct_density"]
feature_df["max_pmi_pos"] = df_train["analysis_tokens"].map(lambda tokens: doc_pmi_score(tokens, "positive"))
feature_df["max_pmi_neg"] = df_train["analysis_tokens"].map(lambda tokens: doc_pmi_score(tokens, "negative"))
feature_df["pmi_gap"] = feature_df["max_pmi_pos"] - feature_df["max_pmi_neg"]

tfidf = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, min_df=10, max_features=5000)
tfidf_matrix = tfidf.fit_transform(df_train["analysis_text"])
svd = TruncatedSVD(n_components=5, random_state=RANDOM_STATE)
svd_matrix = svd.fit_transform(tfidf_matrix)
for idx in range(svd_matrix.shape[1]):
    feature_df[f"svd_{idx+1}"] = svd_matrix[:, idx]

feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
y = df_train["sentiment_name"].map({"negative": 0, "neutral": 1, "positive": 2}).to_numpy()

mi = mutual_info_classif(feature_df, y, random_state=RANDOM_STATE)
mi_table = pd.DataFrame({"feature": feature_df.columns, "mi": mi}).sort_values("mi", ascending=False)


# %% [code cell 32]
fig, ax = plt.subplots(figsize=(9, 7))
top_mi = mi_table.head(10).sort_values("mi", ascending=True)
bars = ax.barh(top_mi["feature"], top_mi["mi"], color="#2d6cdf")
ax.set_title("Top feature theo Mutual Information", loc="left")
ax.set_xlabel("MI score")
ax.set_ylabel("")
ax.grid(axis="x", color="#dfe3eb", linewidth=0.8, alpha=0.8)
sns.despine(ax=ax)
for bar, value in zip(bars, top_mi["mi"]):
    ax.text(bar.get_width() + top_mi["mi"].max() * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
show(fig)


# %% [code cell 33]
from matplotlib.patches import Patch

plot_df = feature_df.assign(sentiment=df_train["sentiment_name"])
feature_specs = [
    ("text_length", "Độ dài ký tự"),
    ("word_count", "Số từ"),
    ("avg_word_length", "Độ dài từ trung bình"),
    ("unique_word_ratio", "Tỷ lệ từ duy nhất"),
    ("aspect_count", "Số aspect"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for ax, (feature, title) in zip(axes, feature_specs):
    sns.kdeplot(
        data=plot_df,
        x=feature,
        hue="sentiment",
        hue_order=SENTIMENT_ORDER,
        palette=SENTIMENT_COLORS,
        common_norm=False,
        fill=True,
        alpha=0.22,
        legend=False,
        ax=ax,
    )
    ax.set_title(title, loc="left")
    ax.set_xlabel(title)
    ax.set_ylabel("Mật độ")
    ax.grid(axis="y", color="#dfe3eb", linewidth=0.8, alpha=0.8)

for ax in axes[len(feature_specs):]:
    ax.axis("off")

handles = [Patch(facecolor=SENTIMENT_COLORS[s], alpha=0.35, label=SENTIMENT_LABELS[s]) for s in SENTIMENT_ORDER]
fig.legend(handles=handles, title="Sentiment", loc="lower center", ncol=3, frameon=False)
fig.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.25)
show(fig)


# %% [code cell 34]
tfidf_svd = pd.DataFrame(svd_matrix, columns=[f"SVD {i+1}" for i in range(svd_matrix.shape[1])])
explained = np.cumsum(svd.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(8.8, 5.2))
ax.plot(np.arange(1, len(explained) + 1), explained, marker="o", color="#2d6cdf")
ax.axhline(0.95, color="#d64b4b", linestyle="--", linewidth=1)
ax.set_title("Cumulative variance của TF-IDF/SVD", loc="left")
ax.set_xlabel("Số chiều")
ax.set_ylabel("Tỷ lệ giải thích")
ax.grid(axis="y", color="#dfe3eb", linewidth=0.8, alpha=0.8)
show(fig)


# %% [code cell 35]
boundary_features = mi_table.head(2)["feature"].tolist()
X2 = StandardScaler().fit_transform(feature_df[boundary_features])
clf = LogisticRegression(max_iter=2000)
clf.fit(X2, y)

x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid = np.c_[xx.ravel(), yy.ravel()]
preds = clf.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, preds, alpha=0.18, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#d64b4b", "#f0ad4e", "#2ca02c"])
sns.scatterplot(
    x=X2[:, 0],
    y=X2[:, 1],
    hue=df_train["sentiment_name"],
    hue_order=SENTIMENT_ORDER,
    palette=SENTIMENT_COLORS,
    alpha=0.75,
    s=35,
    linewidth=0,
    ax=ax,
)
ax.set_title(f"Decision boundary trên 2 feature mạnh nhất: {boundary_features[0]} vs {boundary_features[1]}", loc="left")
ax.set_xlabel(boundary_features[0])
ax.set_ylabel(boundary_features[1])
ax.legend(title="Sentiment", loc="upper right")
show(fig)
