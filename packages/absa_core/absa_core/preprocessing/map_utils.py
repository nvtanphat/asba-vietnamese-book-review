from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping

# Maps are packaged with absa_core so train, API and Docker use identical normalization.
_PACKAGE_MAP_DIR = Path(__file__).resolve().parents[1] / "data" / "maps"
_REPO_MAP_DIR = Path(__file__).resolve().parents[4] / "data" / "maps" if len(Path(__file__).resolve().parents) > 4 else None


def _candidate_dirs() -> list[Path]:
    dirs=[]
    env=os.getenv("ABSA_MAP_DIR")
    if env: dirs.append(Path(env))
    dirs.append(_PACKAGE_MAP_DIR)
    if _REPO_MAP_DIR is not None: dirs.append(_REPO_MAP_DIR)
    dirs.append(Path.cwd()/"data"/"maps")
    seen=[]
    for d in dirs:
        if d not in seen: seen.append(d)
    return seen


@lru_cache(maxsize=None)
def _read_map(path_str: str) -> dict[str, str]:
    path=Path(path_str)
    if not path.exists(): return {}
    with path.open("r",encoding="utf-8") as handle: data=json.load(handle)
    if not isinstance(data,dict): raise ValueError(f"Map file must contain a JSON object: {path}")
    return {str(k):str(v) for k,v in data.items()}


def load_json_map(filename: str, defaults: Mapping[str, str] | None = None) -> dict[str, str]:
    mapping=dict(defaults or {})
    for directory in _candidate_dirs():
        path=directory/filename
        if path.exists():
            mapping.update(_read_map(str(path)))
            break
    return mapping
