from __future__ import annotations
import json
from pathlib import Path
import yaml


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def dump_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
