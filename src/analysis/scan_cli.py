from __future__ import annotations

import argparse
from pathlib import Path

from .data_scanner import DataScanner
from .scan_constants import DEFAULT_OUTPUT_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preprocessing data checks and save a JSON report.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV/JSON file or folder containing the dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON report path.",
    )
    return parser


def run(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    scanner = DataScanner.from_path(args.input)
    report = scanner.run()
    return scanner.save(args.output, report)


def main(argv: list[str] | None = None) -> int:
    output = run(argv)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

