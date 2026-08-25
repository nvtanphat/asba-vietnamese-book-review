from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, environment_info, git_sha, sha256_file, stable_hash, utc_now


def dataset_snapshot(
    data_path: str | Path,
    *,
    root: str | Path = ".",
    split_manifest: str | Path | None = "data/splits/split_manifest.json",
    include_environment: bool = True,
) -> dict[str, Any]:
    data = Path(data_path)
    if not data.exists():
        raise FileNotFoundError(data)
    root = Path(root).resolve()
    payload: dict[str, Any] = {
        "created_at": utc_now(),
        "dataset": {
            "path": data.as_posix(),
            "bytes": data.stat().st_size,
            "sha256": sha256_file(data),
        },
        "git_sha": git_sha(root),
    }
    if split_manifest:
        split = Path(split_manifest)
        if split.exists():
            payload["split_manifest"] = {
                "path": split.as_posix(),
                "sha256": sha256_file(split),
            }
    if include_environment:
        payload["environment"] = environment_info()
    # git_sha changes on every commit regardless of whether the dataset itself changed, so
    # it must stay out of the fingerprint — otherwise `snapshot-data --check` (see
    # mlops/cli.py's cmd_snapshot) would fail on every single commit after the snapshot was
    # recorded even when the data is byte-identical. It's still recorded in the payload for
    # provenance, just not hashed into the identity check.
    payload["fingerprint"] = stable_hash({k: v for k, v in payload.items() if k not in {"created_at", "environment", "git_sha"}})
    return payload


def write_dataset_snapshot(
    data_path: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = dataset_snapshot(data_path, **kwargs)
    atomic_write_json(output_path, payload)
    return payload
