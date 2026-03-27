from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.pipeline import preprocess_file


CLEAN_COLUMNS = [
    'review_id',
    'content',
    'sentiment_llm',
    'as_content',
    'as_physical',
    'as_price',
    'as_packaging',
    'as_delivery',
    'as_service',
]
DEFAULT_INPUTS = {
    'train': Path('data/interim/raw_train/train.json'),
    'val': Path('data/interim/raw_val/val.json'),
    'test': Path('data/interim/raw_test/test.json'),
}
DEFAULT_OUTPUTS = {
    'train': Path('data/processed/train_clean.json'),
    'val': Path('data/processed/val_clean.json'),
    'test': Path('data/processed/test_clean.json'),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Clean raw train/val/test splits into multitask-ready processed outputs.')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--text-column', default='content')
    parser.add_argument('--min-chars', type=int, default=10)
    parser.add_argument('--drop-duplicates', action='store_true', default=True)
    parser.add_argument('--no-drop-duplicates', dest='drop_duplicates', action='store_false')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input) if args.input else DEFAULT_INPUTS[args.split]
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUTS[args.split]
    cleaned = preprocess_file(
        input_path,
        output_path,
        text_column=args.text_column,
        keep_raw=False,
        min_chars=args.min_chars,
        drop_duplicates=args.drop_duplicates,
        keep_columns=CLEAN_COLUMNS,
    )
    print(f'split={args.split}')
    print(f'rows={len(cleaned)}')
    print(output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
