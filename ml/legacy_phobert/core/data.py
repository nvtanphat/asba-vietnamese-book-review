"""Data loading, preparation, and tokenization."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset

from .config import (
    ABSA_PROMPT_PREFIX, ASPECT_COLS, DEFAULT_DATA_ROOT,
    MAX_LENGTH, SENTIMENT_LABELS, TARGET_COLS,
)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    text_col = "text" if "text" in df.columns else "content"
    df = df[[text_col] + TARGET_COLS].dropna(subset=[text_col, "sentiment"]).copy()
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].ne("")]
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)
    df = df[df["sentiment"].isin([0, 1, 2])]
    for col in ASPECT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(3).astype(int)
        df[col] = df[col].apply(lambda x: x if x in [0, 1, 2, 3] else 3)
    return df.reset_index(drop=True)


def load_splits(data_root: Path = DEFAULT_DATA_ROOT):
    train_df = prepare_df(pd.read_json(data_root / "train_clean.json"))
    val_df   = prepare_df(pd.read_json(data_root / "val_clean.json"))
    test_df  = prepare_df(pd.read_json(data_root / "test_clean.json"))
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    dist = train_df["sentiment"].value_counts().sort_index().rename(index=SENTIMENT_LABELS)
    print("Sentiment (train):\n", dist.to_string())
    return train_df, val_df, test_df


def build_absa_input(text: str) -> str:
    return f"{ABSA_PROMPT_PREFIX}{str(text).strip()}"


def tokenize_frame(frame: pd.DataFrame, tokenizer) -> Dataset:
    def _tok(examples):
        texts     = [build_absa_input(t) for t in examples["text"]]
        enc       = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)
        enc["labels"] = [[examples[col][i] for col in TARGET_COLS] for i in range(len(texts))]
        return enc
    ds = Dataset.from_pandas(frame.reset_index(drop=True)).map(
        _tok, batched=True, remove_columns=["text"] + TARGET_COLS)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds
