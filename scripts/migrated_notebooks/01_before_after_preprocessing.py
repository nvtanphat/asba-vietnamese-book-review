"""Migrated from 01_before_after_preprocessing.ipynb.

This file preserves the original experiment code for audit/reproducibility.
The production benchmark uses the modular ml/ package instead.
"""


# %% [code cell 1]
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
    }
)

SENTIMENT_MAP = {0: "Tiếc cực", 1: "Trung lập", 2: "Tích cực"}
SENTIMENT_COLORS = {0: "#d64b4b", 1: "#f0ad4e", 2: "#2ca02c"}
STAGE_ORDER = ["raw", "clean"]
STAGE_COLORS = {"raw": "#7aa6ff", "clean": "#2d6cdf"}
ISSUE_ORDER = ["mã hóa", "noise", "emoji", "trùng lặp"]

def find_root(start=None):
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "README.md").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy project root")

ROOT = find_root()
RAW_TRAIN = ROOT / "data/interim/raw_train/train.json"
CLEAN_TRAIN = ROOT / "data/processed/train_clean.json"
RAW_SCAN = ROOT / "experiments/reports/train_scan.json"
CLEAN_SCAN = ROOT / "experiments/reports/train_clean_scan.json"

def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def scan_metrics(report):
    checks = report["checks"]
    return {
        "số dòng": report["metadata"]["row_count"],
        "content thiếu": checks["missing_values"]["per_column"]["content"]["missing_count"],
        "text quá ngắn": checks["length"]["shorter_than_min_length"],
        "dòng lỗi encoding": checks["encoding"]["rows_with_any_issue"],
        "dòng có noise": checks["noise_patterns"]["rows_with_any_noise"],
        "dòng có emoji": checks["emoji"]["rows_with_emoji"],
        "dòng trùng sau chuẩn hóa": checks["duplicates"]["normalized_duplicate_texts"],
    }

def build_tables(raw_df, clean_df, raw_report, clean_report):
    summary = pd.DataFrame(
        [scan_metrics(raw_report), scan_metrics(clean_report)],
        index=["raw", "clean"],
    ).T
    summary.index.name = "chỉ số"
    summary["delta"] = summary["raw"] - summary["clean"]
    summary["drop_pct"] = (summary["delta"] / summary["raw"] * 100).round(2)

    issue_long = (
        pd.DataFrame(
            {
                "raw": [
                    raw_report["checks"]["encoding"]["rows_with_any_issue"],
                    raw_report["checks"]["noise_patterns"]["rows_with_any_noise"],
                    raw_report["checks"]["emoji"]["rows_with_emoji"],
                    raw_report["checks"]["duplicates"]["normalized_duplicate_texts"],
                ],
                "clean": [
                    clean_report["checks"]["encoding"]["rows_with_any_issue"],
                    clean_report["checks"]["noise_patterns"]["rows_with_any_noise"],
                    clean_report["checks"]["emoji"]["rows_with_emoji"],
                    clean_report["checks"]["duplicates"]["normalized_duplicate_texts"],
                ],
            },
            index=ISSUE_ORDER,
        )
        .rename_axis("vấn đề")
        .reset_index()
        .melt(id_vars="vấn đề", var_name="stage", value_name="count")
    )

    label_compare = pd.DataFrame(
        {
            "raw": raw_df["sentiment"].value_counts().sort_index(),
            "clean": clean_df["sentiment"].value_counts().sort_index(),
        }
    ).fillna(0).astype(int)
    label_compare.index = label_compare.index.map(lambda x: SENTIMENT_MAP.get(int(x), str(x)))
    label_compare.index.name = "nhãn"
    label_long = label_compare.reset_index().melt(id_vars="nhãn", var_name="stage", value_name="count")

    length_long = pd.DataFrame(
        {
            "raw": raw_df["content"].fillna("").str.split().map(len),
            "clean": clean_df["content"].fillna("").str.split().map(len),
        }
    ).melt(var_name="stage", value_name="words")
    return summary, issue_long, label_compare, label_long, length_long

def build_changed_frame(raw_df, clean_df):
    raw_lookup = raw_df[["review_id", "content", "product_name"]].rename(columns={"content": "content_raw"})
    aligned = clean_df.rename(columns={"content": "content_clean"}).merge(
        raw_lookup,
        on="review_id",
        how="left",
        validate="one_to_one",
    )
    changed = aligned.loc[aligned["content_raw"].fillna("") != aligned["content_clean"].fillna("")].copy()
    changed["sentiment_name"] = changed["sentiment"].map(
        lambda x: SENTIMENT_MAP.get(int(x), str(x)) if pd.notna(x) else str(x)
    )
    changed["raw_len"] = changed["content_raw"].fillna("").str.len()
    changed["clean_len"] = changed["content_clean"].fillna("").str.len()
    changed["delta"] = changed["raw_len"] - changed["clean_len"]
    changed["word_delta"] = (
        changed["content_raw"].fillna("").str.split().map(len)
        - changed["content_clean"].fillna("").str.split().map(len)
    )
    return changed

def style(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#dfe3eb", linewidth=0.8, alpha=0.9)
    sns.despine(ax=ax)

def bar_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)

def plot_overview(issue_long, label_long, length_long, changed):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("So sánh trước và sau tiền xử lý", x=0.01, ha="left", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    sns.barplot(
        data=issue_long,
        x="vấn đề",
        y="count",
        hue="stage",
        order=ISSUE_ORDER,
        hue_order=STAGE_ORDER,
        palette=STAGE_COLORS,
        errorbar=None,
        ax=ax,
    )
    style(ax)
    ax.set_title("Số lỗi trước và sau", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("số dòng")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title=None, frameon=False)
    bar_labels(ax)

    ax = axes[0, 1]
    sns.histplot(
        data=length_long,
        x="words",
        hue="stage",
        bins=35,
        stat="count",
        common_norm=False,
        multiple="layer",
        alpha=0.5,
        palette=STAGE_COLORS,
        ax=ax,
    )
    style(ax)
    ax.set_title("Phân bố số từ", loc="left")
    ax.set_xlabel("số từ")
    ax.set_ylabel("số dòng")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title=None, frameon=False)

    ax = axes[1, 0]
    sns.barplot(
        data=label_long,
        x="nhãn",
        y="count",
        hue="stage",
        order=list(SENTIMENT_MAP.values()),
        hue_order=STAGE_ORDER,
        palette=STAGE_COLORS,
        errorbar=None,
        ax=ax,
    )
    style(ax)
    ax.set_title("Phân bố nhãn sentiment", loc="left")
    ax.set_xlabel("nhãn")
    ax.set_ylabel("số dòng")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title=None, frameon=False)
    bar_labels(ax)

    ax = axes[1, 1]
    sns.histplot(data=changed, x="delta", bins=35, color="#1f4e79", ax=ax)
    style(ax)
    ax.set_title("Phân bố chênh lệch độ dài", loc="left")
    ax.set_xlabel("chênh lệch ký tự (thô - sạch)")
    ax.set_ylabel("số dòng")

    return fig

def plot_changed_scatter(changed, sample_size=800, random_state=42):
    sample = changed.sample(min(len(changed), sample_size), random_state=random_state) if not changed.empty else changed.copy()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    if not sample.empty:
        sns.scatterplot(
            data=sample,
            x="raw_len",
            y="clean_len",
            hue="sentiment_name",
            palette={SENTIMENT_MAP[k]: SENTIMENT_COLORS[k] for k in SENTIMENT_MAP},
            s=24,
            alpha=0.55,
            edgecolor="none",
            ax=ax,
        )
    max_len = max(sample["raw_len"].max(), sample["clean_len"].max()) if not sample.empty else 1
    ax.plot([0, max_len], [0, max_len], linestyle="--", linewidth=1.2, color="#666666", alpha=0.7)
    style(ax, grid_axis="both")
    ax.set_title("Độ dài thô và sạch trên các dòng thay đổi", loc="left")
    ax.set_xlabel("số ký tự thô")
    ax.set_ylabel("số ký tự sạch")
    ax.set_xlim(0, max_len * 1.02 if max_len else 1)
    ax.set_ylim(0, max_len * 1.02 if max_len else 1)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title="nhãn cảm xúc", frameon=False)
    return fig

def top_examples(changed, limit=5):
    top_changed = changed.sort_values("delta", ascending=False).head(limit)
    product_summary = (
        changed.groupby("product_name", dropna=False)
        .agg(rows=("review_id", "count"), avg_delta=("delta", "mean"), avg_words_delta=("word_delta", "mean"))
        .sort_values(["rows", "avg_delta"], ascending=[False, False])
        .head(limit)
        .reset_index()
    )
    return top_changed, product_summary


# %% [code cell 2]
raw_df = pd.read_json(RAW_TRAIN)
clean_df = pd.read_json(CLEAN_TRAIN)
raw_report = read_json(RAW_SCAN)
clean_report = read_json(CLEAN_SCAN)

summary, issue_long, label_compare, label_long, length_long = build_tables(
    raw_df, clean_df, raw_report, clean_report
)
changed = build_changed_frame(raw_df, clean_df)
top_changed, product_summary = top_examples(changed)

print(f"Số dòng raw: {len(raw_df)}")
print(f"Số dòng clean: {len(clean_df)}")
print(f"Số dòng bị loại: {len(raw_df) - len(clean_df)}")
print(f"Số dòng có content thay đổi sau clean: {len(changed)}")
print(f"Tỉ lệ dòng thay đổi trên clean: {len(changed) / len(clean_df):.2%}")
print(f"review_id raw duy nhất: {raw_df['review_id'].is_unique}")
print(f"review_id clean duy nhất: {clean_df['review_id'].is_unique}")
print(f"Các cỗt clean: {list(clean_df.columns)}")

display(summary)
display(label_compare)


# %% [code cell 3]
fig = plot_overview(issue_long, label_long, length_long, changed)
display(fig)
plt.close(fig)


# %% [code cell 4]
with pd.option_context("display.max_colwidth", 120):
    display(
        top_changed[[
            "review_id",
            "product_name",
            "sentiment_name",
            "raw_len",
            "clean_len",
            "delta",
            "content_raw",
            "content_clean",
        ]]
    )

fig = plot_changed_scatter(changed)
display(fig)
plt.close(fig)

display(product_summary)
