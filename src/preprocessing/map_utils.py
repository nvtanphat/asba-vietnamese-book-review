from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = PROJECT_ROOT / "data" / "maps"


@lru_cache(maxsize=None)
def _read_map(path_str: str) -> dict[str, str]:
    path = Path(path_str)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Map file must contain an object: {path}")
    return {str(key): str(value) for key, value in data.items()}


def load_json_map(filename: str, defaults: Mapping[str, str] | None = None) -> dict[str, str]:
    mapping = dict(defaults or {})
    mapping.update(_read_map(str(MAP_DIR / filename)))
    return mapping
