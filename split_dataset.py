from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = Path("data/raw/tiki-book-review.csv")
TRAIN_OUT = Path("data/interim/train/train.json")
TEST_OUT = Path("data/interim/test/test.json")
LABEL = "sentiment_llm"


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df[df[LABEL].notna()].copy()

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[LABEL],
    )

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_json(TRAIN_OUT, orient="records", force_ascii=False, indent=2)
    test.to_json(TEST_OUT, orient="records", force_ascii=False, indent=2)

    print(f"train={len(train)}")
    print(f"test={len(test)}")
    print(TRAIN_OUT)
    print(TEST_OUT)


if __name__ == "__main__":
    main()
