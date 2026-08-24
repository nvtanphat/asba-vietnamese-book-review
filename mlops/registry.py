from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .common import atomic_write_json, read_json, stable_hash, utc_now
from .gates import evaluate_quality_gate


class LocalModelRegistry:
    """Git/Kaggle-friendly model registry with immutable versions + mutable aliases."""

    def __init__(self, path: str | Path = "artifacts/registry/registry.json", root: str | Path = ".") -> None:
        self.root = Path(root)
        self.path = self.root / path if not Path(path).is_absolute() else Path(path)
        self.data = read_json(self.path, {"schema_version": 1, "models": {}, "aliases": {}})

    def _save(self) -> None:
        atomic_write_json(self.path, self.data)

    def register(
        self,
        *,
        model: str,
        artifact_dir: str | Path,
        metrics: dict[str, Any],
        lineage: dict[str, Any] | None = None,
        tracking_run_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        source = Path(artifact_dir)
        if not source.exists():
            raise FileNotFoundError(source)
        identity = {
            "model": model,
            "metrics": metrics,
            "data_fingerprint": (lineage or {}).get("fingerprint"),
            "tracking_run_id": tracking_run_id,
        }
        version = f"{utc_now().replace(':','').replace('-','')[:15]}-{stable_hash(identity)[:8]}"
        entry = {
            "model": model,
            "version": version,
            "created_at": utc_now(),
            "artifact_dir": source.as_posix(),
            "metrics": metrics,
            "lineage": lineage or {},
            "tracking_run_id": tracking_run_id,
            "notes": notes,
        }
        versions = self.data.setdefault("models", {}).setdefault(model, [])
        if not any(v["version"] == version for v in versions):
            versions.append(entry)
        self.data.setdefault("aliases", {})["candidate"] = {"model": model, "version": version, "updated_at": utc_now()}
        self._save()
        return entry

    def get(self, model: str, version: str) -> dict[str, Any]:
        for item in self.data.get("models", {}).get(model, []):
            if item.get("version") == version:
                return item
        raise KeyError(f"Unknown model version: {model}@{version}")

    def resolve_alias(self, alias: str) -> dict[str, Any] | None:
        ref = self.data.get("aliases", {}).get(alias)
        return None if not ref else self.get(ref["model"], ref["version"])

    def promote(
        self,
        *,
        model: str,
        version: str,
        alias: str,
        gate: dict[str, Any] | None = None,
        release_root: str | Path | None = None,
        production_artifact: str | Path | None = None,
    ) -> dict[str, Any]:
        entry = self.get(model, version)
        gate_result = evaluate_quality_gate(entry["metrics"], gate or {})
        if not gate_result["passed"]:
            raise RuntimeError(f"Quality gate failed for {model}@{version}: {json.dumps(gate_result, ensure_ascii=False)}")

        if release_root:
            release_dir = self.root / release_root / f"{model}-{version}"
            source = Path(entry["artifact_dir"])
            if release_dir.exists():
                shutil.rmtree(release_dir)
            shutil.copytree(source, release_dir)
            entry["release_dir"] = release_dir.as_posix()

        if alias == "champion" and production_artifact:
            source = Path(entry.get("release_dir") or entry["artifact_dir"])
            dst = self.root / production_artifact
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(source, dst)

        self.data.setdefault("aliases", {})[alias] = {"model": model, "version": version, "updated_at": utc_now()}
        entry.setdefault("promotions", []).append({"alias": alias, "at": utc_now(), "gate": gate_result})
        self._save()
        return {"entry": entry, "gate": gate_result, "alias": alias}

    def summary(self) -> dict[str, Any]:
        return self.data
