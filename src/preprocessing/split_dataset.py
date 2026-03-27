from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import quality_filter
from src.preprocessing.pipeline import clean_text_series, preprocess_dataframe

CSV_ENCODING = "utf-8-sig"
JSON_ENCODING = "utf-8"

INPUT_CANDIDATES = (
    Path("data/raw/tiki-book-review.json"),
    Path("data/raw/tiki-book-review.csv"),
)
INPUT = next((path for path in INPUT_CANDIDATES if path.exists()), INPUT_CANDIDATES[0])

RAW_INTERIM_TRAIN_OUT = Path("data/interim/raw_train/train.json")
RAW_INTERIM_VAL_OUT = Path("data/interim/raw_val/val.json")
RAW_INTERIM_TEST_OUT = Path("data/interim/raw_test/test.json")
PROCESSED_TRAIN_OUT = Path("data/processed/train_clean.json")
PROCESSED_VAL_OUT = Path("data/processed/val_clean.json")
PROCESSED_TEST_OUT = Path("data/processed/test_clean.json")

LABEL = "sentiment_llm"
TEXT_COLUMN = "content"
VAL_SIZE = 0.15
TEST_SIZE = 0.15
HOLDOUT_SIZE = VAL_SIZE + TEST_SIZE
VAL_SHARE_IN_HOLDOUT = VAL_SIZE / HOLDOUT_SIZE
MIN_CHARS = quality_filter.SHORT_TEXT_MIN_CHARS

CLEAN_COLUMNS = [
    "review_id",
    "content",
    "sentiment_llm",
    "as_content",
    "as_physical",
    "as_price",
    "as_packaging",
    "as_delivery",
    "as_service",
]
RANDOM_STATE = 42


def _write_split(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=JSON_ENCODING, newline="") as handle:
        frame.to_json(handle, orient="records", force_ascii=False, indent=2)


def _load_labeled_rows() -> pd.DataFrame:
    if INPUT.suffix.lower() == ".json":
        with INPUT.open("r", encoding=JSON_ENCODING) as handle:
            frame = pd.read_json(handle)
    else:
        frame = pd.read_csv(INPUT, encoding=CSV_ENCODING)
    frame = frame[frame[LABEL].notna()].copy()
    frame[LABEL] = pd.to_numeric(frame[LABEL], errors="coerce")
    frame = frame[frame[LABEL].isin([0, 1, 2])].copy()
    frame[LABEL] = frame[LABEL].astype(int)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and int(counts.min()) >= 2


def _split_group_table(groups: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = groups["dominant_label"] if _can_stratify(groups["dominant_label"]) else None
    left, right = train_test_split(
        groups,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return left.reset_index(drop=True), right.reset_index(drop=True)


def _split_raw_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = frame.copy()
    cleaned_text = clean_text_series(grouped[TEXT_COLUMN])
    grouped["__group_key__"] = cleaned_text.map(quality_filter.normalize_for_duplicate)

    group_table = (
        grouped.groupby("__group_key__", dropna=False, sort=False)
        .agg(
            dominant_label=(LABEL, lambda values: values.value_counts().idxmax()),
        )
        .reset_index()
    )

    train_groups, holdout_groups = _split_group_table(
        group_table,
        test_size=HOLDOUT_SIZE,
        seed=RANDOM_STATE,
    )
    val_groups, test_groups = _split_group_table(
        holdout_groups,
        test_size=1 - VAL_SHARE_IN_HOLDOUT,
        seed=RANDOM_STATE + 1,
    )

    train_keys = set(train_groups["__group_key__"])
    val_keys = set(val_groups["__group_key__"])
    test_keys = set(test_groups["__group_key__"])

    raw_train = grouped[grouped["__group_key__"].isin(train_keys)].drop(columns=["__group_key__"])
    raw_val = grouped[grouped["__group_key__"].isin(val_keys)].drop(columns=["__group_key__"])
    raw_test = grouped[grouped["__group_key__"].isin(test_keys)].drop(columns=["__group_key__"])

    return (
        raw_train.reset_index(drop=True),
        raw_val.reset_index(drop=True),
        raw_test.reset_index(drop=True),
    )


def _clean_split(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = preprocess_dataframe(
        frame,
        text_column=TEXT_COLUMN,
        keep_raw=False,
        min_chars=MIN_CHARS,
        drop_duplicates=False,
        keep_columns=CLEAN_COLUMNS,
    )
    cleaned[LABEL] = pd.to_numeric(cleaned[LABEL], errors="coerce").astype(int)
    return cleaned


def main() -> None:
    labeled = _load_labeled_rows()
    raw_train, raw_val, raw_test = _split_raw_rows(labeled)

    train = _clean_split(raw_train)
    val = _clean_split(raw_val)
    test = _clean_split(raw_test)

    _write_split(raw_train, RAW_INTERIM_TRAIN_OUT)
    _write_split(raw_val, RAW_INTERIM_VAL_OUT)
    _write_split(raw_test, RAW_INTERIM_TEST_OUT)
    _write_split(train, PROCESSED_TRAIN_OUT)
    _write_split(val, PROCESSED_VAL_OUT)
    _write_split(test, PROCESSED_TEST_OUT)

    print(f"labeled_rows={len(labeled)}")
    print(f"raw_train={len(raw_train)}")
    print(f"raw_val={len(raw_val)}")
    print(f"raw_test={len(raw_test)}")
    print(f"train={len(train)}")
    print(f"val={len(val)}")
    print(f"test={len(test)}")
    print(RAW_INTERIM_TRAIN_OUT)
    print(RAW_INTERIM_VAL_OUT)
    print(RAW_INTERIM_TEST_OUT)
    print(PROCESSED_TRAIN_OUT)
    print(PROCESSED_VAL_OUT)
    print(PROCESSED_TEST_OUT)


if __name__ == "__main__":
    main()
