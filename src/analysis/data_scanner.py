from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import (
    duplicate_check,
    emoji_check,
    encoding_check,
    label_distribution_check,
    length_check,
    missing_values_check,
    noise_pattern_check,
    overview_check,
    vocab_check,
)
from .helpers import detect_text_column
from .scan_constants import DEFAULT_INPUT_CANDIDATES

CHECKS = (
    ("overview", overview_check.scan),
    ("missing_values", missing_values_check.scan),
    ("length", length_check.scan),
    ("encoding", encoding_check.scan),
    ("noise_patterns", noise_pattern_check.scan),
    ("emoji", emoji_check.scan),
    ("vocab", vocab_check.scan),
    ("duplicates", duplicate_check.scan),
    ("labels", label_distribution_check.scan),
)


def resolve_input_path(input_path: str | Path | None = None) -> Path:
    if input_path is not None:
        path = Path(input_path)
        if path.is_dir():
            for candidate in DEFAULT_INPUT_CANDIDATES:
                candidate_path = path / candidate.name
                if candidate_path.exists():
                    return candidate_path
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[0]


def _read_text_file(path: Path, encoding: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        frame = pd.read_json(path, encoding=encoding)
        return frame.to_dict(orient="records")

    if suffix == ".csv":
        frame = pd.read_csv(path, encoding=encoding)
        return frame.to_dict(orient="records")

    raise ValueError(f"Unsupported input format: {suffix}")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
        try:
            return _read_text_file(source, encoding)
        except Exception as exc:  # pragma: no cover - fallback path
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(f"Unable to read {source}. Tried: {'; '.join(errors)}")


class DataScanner:
    def __init__(self, records: Sequence[Mapping[str, Any]], source_path: str | Path | None = None) -> None:
        self.records = [dict(row) for row in records]
        self.source_path = str(source_path) if source_path is not None else None
        self.text_column = detect_text_column(self.records)

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> "DataScanner":
        resolved = resolve_input_path(path)
        return cls(load_records(resolved), resolved)

    def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "metadata": {
                "source_path": self.source_path,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "row_count": len(self.records),
                "column_count": len(self.records[0]) if self.records else 0,
                "text_column": self.text_column,
            },
            "checks": {},
        }

        for name, check_fn in CHECKS:
            report["checks"][name] = check_fn(self.records, self.text_column)

        return report

    def save(self, output_path: str | Path, report: dict[str, Any] | None = None) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = report if report is not None else self.run()
        with output.open("w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return output


def scan_records(records: Sequence[Mapping[str, Any]], source_path: str | Path | None = None) -> dict[str, Any]:
    return DataScanner(records, source_path).run()


def scan_path(path: str | Path | None = None) -> dict[str, Any]:
    return DataScanner.from_path(path).run()
