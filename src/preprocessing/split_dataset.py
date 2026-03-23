from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.pipeline import preprocess_dataframe

INPUT = Path("data/raw/tiki-book-review.csv")
INTERIM_TRAIN_OUT = Path("data/interim/train/train.json")
INTERIM_TEST_OUT = Path("data/interim/test/test.json")
PROCESSED_TRAIN_OUT = Path("data/processed/train_clean.json")
PROCESSED_TEST_OUT = Path("data/processed/test_clean.json")
LABEL = "sentiment_llm"
TEXT_COLUMN = "content"
TEST_SIZE = 0.2

KEEP_COLUMNS = [
    'review_id',
    'rating',
    'review_title',
    'product_name',
    'category',
    'content',
    'content_raw',
    'sentiment_llm',
    'as_content',
    'as_physical',
    'as_price',
    'as_packaging',
    'as_delivery',
    'as_service',
]
RANDOM_STATE = 42


def _write_split(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(output_path, orient="records", force_ascii=False, indent=2)


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df[df[LABEL].notna()].copy()
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
    df = df[df[LABEL].isin([0, 1, 2])].copy()

    # Clean the full corpus before splitting so duplicate normalized texts
    # cannot leak across train/test.
    cleaned = preprocess_dataframe(
        df,
        text_column=TEXT_COLUMN,
        keep_raw=True,
        min_chars=10,
        drop_duplicates=True,
        keep_columns=KEEP_COLUMNS,
    )
    cleaned[LABEL] = pd.to_numeric(cleaned[LABEL], errors="coerce").astype(int)

    train, test = train_test_split(
        cleaned,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=cleaned[LABEL],
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    _write_split(train, INTERIM_TRAIN_OUT)
    _write_split(test, INTERIM_TEST_OUT)
    _write_split(train, PROCESSED_TRAIN_OUT)
    _write_split(test, PROCESSED_TEST_OUT)

    print(f"train={len(train)}")
    print(f"test={len(test)}")
    print(INTERIM_TRAIN_OUT)
    print(INTERIM_TEST_OUT)
    print(PROCESSED_TRAIN_OUT)
    print(PROCESSED_TEST_OUT)


if __name__ == "__main__":
    main()
