from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.pipeline import preprocess_file


DEFAULT_INPUT = Path("data/interim/train/train.json")
DEFAULT_OUTPUT = Path("data/processed/train_clean.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean train split review text.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--keep-raw", action="store_true", default=True)
    parser.add_argument("--no-keep-raw", dest="keep_raw", action="store_false")
    parser.add_argument("--drop-duplicates", action="store_true", default=True)
    parser.add_argument("--no-drop-duplicates", dest="drop_duplicates", action="store_false")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cleaned = preprocess_file(
        args.input,
        args.output,
        text_column=args.text_column,
        keep_raw=args.keep_raw,
        min_chars=args.min_chars,
        drop_duplicates=args.drop_duplicates,
    )
    print(f"rows={len(cleaned)}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
