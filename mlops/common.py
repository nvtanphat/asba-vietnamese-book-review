from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def git_sha(root: str | Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True, stderr=subprocess.DEVNULL)
        return out.strip() or None
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "executable": sys.executable,
    }


def flatten_dict(data: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, name, sep))
        elif isinstance(value, (list, tuple)):
            out[name] = ",".join(map(str, value))
        else:
            out[name] = value
    return out
