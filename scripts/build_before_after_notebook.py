from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_Before_After_Preprocessing.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(True),
    }


def build_notebook() -> dict:
    cells = [
        md(
            """
            # 01 - Before / After Preprocessing

            Notebook này so sánh nhanh raw train split với cleaned train split.

            Trọng tâm:
            - giữ ít code hơn
            - rút ra nhiều insight hơn
            - xem được hiệu quả của preprocessing ngay trên text, labels, và noise
            """
        ),
        code(
            """
            from pathlib import Path
            import json

            import matplotlib.pyplot as plt
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from IPython.display import display

            pd.set_option("display.max_colwidth", 140)
            pd.set_option("display.width", 180)

            def find_project_root(start: Path | None = None) -> Path:
                start = (start or Path.cwd()).resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "preprocessing_pipeline.md").exists() and (candidate / "data").exists():
                        return candidate
                raise FileNotFoundError("Cannot locate project root")

            ROOT = find_project_root()
            RAW_TRAIN = ROOT / "data/interim/train/train.json"
            CLEAN_TRAIN = ROOT / "data/processed/train_clean.json"
            RAW_SCAN = ROOT / "experiments/reports/train_scan.json"
            CLEAN_SCAN = ROOT / "experiments/reports/train_clean_scan.json"

            raw_df = pd.read_json(RAW_TRAIN)
            clean_df = pd.read_json(CLEAN_TRAIN)
            raw_report = json.loads(RAW_SCAN.read_text(encoding="utf-8"))
            clean_report = json.loads(CLEAN_SCAN.read_text(encoding="utf-8"))

            print(f"raw rows: {len(raw_df)}")
            print(f"clean rows: {len(clean_df)}")
            print(f"rows removed: {len(raw_df) - len(clean_df)}")
            """
        ),
        code(
            """
            def extract_metrics(report: dict) -> dict:
                checks = report["checks"]
                return {
                    "rows": report["metadata"]["row_count"],
                    "missing_content": checks["missing_values"]["per_column"]["content"]["missing_count"],
                    "short_texts": checks["length"]["shorter_than_min_length"],
                    "encoding_rows": checks["encoding"]["rows_with_any_issue"],
                    "noise_rows": checks["noise_patterns"]["rows_with_any_noise"],
                    "emoji_rows": checks["emoji"]["rows_with_emoji"],
                    "duplicates": checks["duplicates"]["normalized_duplicate_texts"],
                }


            summary = pd.DataFrame(
                [extract_metrics(raw_report), extract_metrics(clean_report)],
                index=["raw", "clean"],
            ).T

            label_before = raw_df["sentiment_llm"].value_counts().sort_index()
            label_after = clean_df["sentiment_llm"].value_counts().sort_index()
            label_compare = pd.DataFrame({"raw": label_before, "clean": label_after}).fillna(0).astype(int)

            raw_len = raw_df["content"].fillna("").str.len()
            clean_len = clean_df["content"].fillna("").str.len()
            raw_words = raw_df["content"].fillna("").str.split().map(len)
            clean_words = clean_df["content"].fillna("").str.split().map(len)

            display(summary)
            display(label_compare)
            """
        ),
        code(
            """
            issue_df = pd.DataFrame(
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
                index=["encoding", "noise", "emoji", "duplicates"],
            )

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Issue counts before vs after",
                    "Word count distribution",
                    "Label distribution",
                    "Length delta distribution",
                ),
            )

            for col, split, color in [(1, "raw", "#7aa6ff"), (1, "clean", "#2d6cdf")]:
                fig.add_trace(
                    go.Bar(name=split, x=issue_df.index, y=issue_df[split], marker_color=color, showlegend=(col == 1)),
                    row=1,
                    col=1,
                )

            fig.add_trace(go.Histogram(x=raw_words, name="raw", opacity=0.55, nbinsx=35, marker_color="#7aa6ff"), row=1, col=2)
            fig.add_trace(go.Histogram(x=clean_words, name="clean", opacity=0.55, nbinsx=35, marker_color="#2d6cdf"), row=1, col=2)

            fig.add_trace(go.Bar(x=label_compare.index.astype(str), y=label_compare["raw"], name="raw labels", marker_color="#7aa6ff"), row=2, col=1)
            fig.add_trace(go.Bar(x=label_compare.index.astype(str), y=label_compare["clean"], name="clean labels", marker_color="#2d6cdf"), row=2, col=1)

            fig.add_trace(go.Histogram(x=changed["delta"], nbinsx=35, marker_color="#1f4e79", showlegend=False), row=2, col=2)

            fig.update_layout(
                template="plotly_white",
                height=900,
                barmode="overlay",
                title="Before vs after preprocessing",
            )
            fig.update_xaxes(title_text="issue type", row=1, col=1)
            fig.update_xaxes(title_text="word count", row=1, col=2)
            fig.update_xaxes(title_text="sentiment_llm", row=2, col=1)
            fig.update_xaxes(title_text="raw char count - clean char count", row=2, col=2)
            fig.update_yaxes(title_text="rows", row=1, col=1)
            fig.update_yaxes(title_text="rows", row=1, col=2)
            fig.update_yaxes(title_text="rows", row=2, col=1)
            fig.update_yaxes(title_text="rows", row=2, col=2)
            fig.show()
            """
        ),
        code(
            """
            changed = clean_df.loc[clean_df["content_raw"].fillna("") != clean_df["content"].fillna("")].copy()
            changed["raw_len"] = changed["content_raw"].fillna("").str.len()
            changed["clean_len"] = changed["content"].fillna("").str.len()
            changed["delta"] = changed["raw_len"] - changed["clean_len"]
            changed["word_delta"] = changed["content_raw"].fillna("").str.split().map(len) - changed["content"].fillna("").str.split().map(len)

            top_changed = changed.sort_values("delta", ascending=False).head(10)
            display(top_changed[["review_id", "product_name", "sentiment_llm", "raw_len", "clean_len", "delta", "content_raw", "content"]])

            fig = px.scatter(
                changed.sample(min(len(changed), 800), random_state=42),
                x="raw_len",
                y="clean_len",
                color="sentiment_llm",
                opacity=0.55,
                title="Raw vs clean length on changed rows",
                labels={"raw_len": "raw char count", "clean_len": "clean char count"},
            )
            fig.update_layout(template="plotly_white")
            fig.show()

            product_summary = (
                changed.groupby("product_name")
                .agg(rows=("review_id", "count"), avg_delta=("delta", "mean"), avg_words_delta=("word_delta", "mean"))
                .sort_values(["rows", "avg_delta"], ascending=[False, False])
                .head(15)
                .reset_index()
            )
            display(product_summary)
            """
        ),
        md(
            """
            ## Takeaways

            - `content` được làm sạch mạnh phần kỹ thuật, nhưng vẫn giữ tín hiệu sentiment như emoji và dấu nhấn.
            - Label distribution gần như giữ nguyên sau clean.
            - `raw_len - clean_len` cho thấy preprocessing chủ yếu dọn noise, không làm mất quá nhiều nội dung.
            - Những product có nhiều review và delta lớn thường là nơi dữ liệu bẩn hoặc có tín hiệu cảm xúc mạnh.
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
