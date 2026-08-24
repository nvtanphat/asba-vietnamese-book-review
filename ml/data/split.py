from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from absa_core.preprocessing.pipeline import clean_text_series, preprocess_dataframe
from absa_core.preprocessing.quality_filter import normalize_for_duplicate, SHORT_TEXT_MIN_CHARS

from .loader import read_frame, prepare_frame
from .schema import ASPECT_COLS
from .validation import write_manifest

SEED = 42


def _stratify_key(grouped: pd.DataFrame) -> pd.Series:
    # Preserve overall sentiment and rough aspect density. Fall back to sentiment if a bucket is too rare.
    key = grouped["dominant_sentiment"].astype(str) + "|" + grouped["aspect_bucket"].astype(str)
    return key if key.value_counts().min() >= 2 else grouped["dominant_sentiment"]


def _text_group_split(frame: pd.DataFrame, val_size: float, test_size: float, seed: int):
    tmp = frame.copy()
    cleaned = clean_text_series(tmp["content"])
    tmp["__group__"] = cleaned.map(normalize_for_duplicate)
    for col in ASPECT_COLS:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(3).astype(int)
    tmp["__present_count__"] = tmp[ASPECT_COLS].ne(3).sum(axis=1)

    groups = (
        tmp.groupby("__group__", sort=False)
        .agg(
            dominant_sentiment=("sentiment", lambda x: int(pd.Series(x).value_counts().idxmax())),
            aspect_count=("__present_count__", "median"),
        )
        .reset_index()
    )
    groups["aspect_bucket"] = pd.cut(groups["aspect_count"], bins=[-1, 0, 1, 2, 6], labels=[0, 1, 2, 3]).astype(int)
    holdout = val_size + test_size
    tr_g, ho_g = train_test_split(groups, test_size=holdout, random_state=seed, stratify=_stratify_key(groups))
    val_share = val_size / holdout
    strat = _stratify_key(ho_g)
    va_g, te_g = train_test_split(ho_g, test_size=1 - val_share, random_state=seed + 1, stratify=strat)

    keys = [set(g["__group__"]) for g in (tr_g, va_g, te_g)]
    return tuple(tmp[tmp["__group__"].isin(k)].drop(columns=["__group__", "__present_count__"]).reset_index(drop=True) for k in keys)


def _product_group_split(frame: pd.DataFrame, val_size: float, test_size: float, seed: int):
    if "product_id" not in frame.columns:
        raise ValueError("product_group strategy requires product_id")
    holdout = val_size + test_size
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    tr_idx, ho_idx = next(gss.split(frame, groups=frame["product_id"]))
    train, hold = frame.iloc[tr_idx].reset_index(drop=True), frame.iloc[ho_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size / holdout, random_state=seed + 1)
    va_idx, te_idx = next(gss2.split(hold, groups=hold["product_id"]))
    return train, hold.iloc[va_idx].reset_index(drop=True), hold.iloc[te_idx].reset_index(drop=True)


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["review_id", "product_id", "product_name", "category", "rating", "created_at", "content", "sentiment", *ASPECT_COLS] if c in frame.columns]
    out = preprocess_dataframe(
        frame,
        text_column="content",
        keep_raw=False,
        min_chars=SHORT_TEXT_MIN_CHARS,
        drop_duplicates=False,
        keep_columns=keep,
        lowercase=True,
    )
    return prepare_frame(out, text_col="content")


def create_splits(input_path: str | Path, output_dir: str | Path = "data/splits", *, val_size: float = 0.15, test_size: float = 0.15, seed: int = SEED, strategy: str = "text_group_stratified"):
    raw = read_frame(input_path)
    raw["sentiment"] = pd.to_numeric(raw["sentiment"], errors="coerce")
    raw = raw[raw["sentiment"].isin([0, 1, 2])].copy()
    raw["sentiment"] = raw["sentiment"].astype(int)

    if strategy == "text_group_stratified":
        raw_splits = _text_group_split(raw, val_size, test_size, seed)
    elif strategy == "product_group":
        raw_splits = _product_group_split(raw, val_size, test_size, seed)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    splits = {name: _clean(df) for name, df in zip(["train", "val", "test"], raw_splits)}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_json(out / f"{name}.json", orient="records", force_ascii=False, indent=2)
    reloaded_splits = {name: prepare_frame(read_frame(out / f"{name}.json")) for name in splits}
    write_manifest(out / "split_manifest.json", seed=seed, strategy=strategy, splits=reloaded_splits)
    return reloaded_splits



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/tiki-book-review_merged_fixed_v3.json")
    p.add_argument("--output-dir", default="data/splits")
    p.add_argument("--strategy", choices=["text_group_stratified", "product_group"], default="text_group_stratified")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    splits = create_splits(args.input, args.output_dir, seed=args.seed, strategy=args.strategy)
    print(" | ".join(f"{k}={len(v):,}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
