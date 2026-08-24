from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_scanner import scan_path, scan_records


def scan_source(source: str | Path | None = None) -> dict[str, Any]:
    return scan_path(source)


def scan_file(path: str | Path) -> dict[str, Any]:
    return scan_path(path)


def scan_rows(rows: Sequence[Mapping[str, Any]], source_path: str | Path | None = None) -> dict[str, Any]:
    return scan_records(rows, source_path)

