from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .common import atomic_write_json, flatten_dict, utc_now


class ExperimentTracker:
    """Small tracking facade.

    Default backend is local JSON so training never depends on an external service.
    Set SENTENAI_TRACKING_BACKEND=mlflow and MLFLOW_TRACKING_URI to use MLflow 3.
    """

    def __init__(
        self,
        *,
        experiment: str,
        run_name: str | None = None,
        root: str | Path = ".",
        backend: str | None = None,
        local_dir: str | Path = "experiments/_tracking",
        tracking_uri: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.experiment = experiment
        self.run_name = run_name or experiment
        self.backend = (backend or os.getenv("SENTENAI_TRACKING_BACKEND", "local")).lower()
        self.local_dir = self.root / local_dir
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.run_id = uuid.uuid4().hex
        self._mlflow = None
        self._active = False
        self._record: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "run_name": self.run_name,
            "status": "CREATED",
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
        }

    @property
    def run_dir(self) -> Path:
        return self.local_dir / self.experiment / self.run_id

    def start(self) -> "ExperimentTracker":
        self._record.update({"status": "RUNNING", "started_at": utc_now()})
        if self.backend == "off":
            self._active = True
            return self
        if self.backend == "mlflow":
            try:
                import mlflow
            except ImportError as exc:
                raise RuntimeError("MLflow backend requested but mlflow is not installed. Install: uv sync --group mlops") from exc
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            run = mlflow.start_run(run_name=self.run_name)
            self.run_id = run.info.run_id
            self._record["run_id"] = self.run_id
            self._mlflow = mlflow
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.run_dir / "run.json", self._record)
        self._active = True
        return self

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: v for k, v in flatten_dict(params).items() if v is not None}
        self._record["params"].update(flat)
        if self._mlflow:
            safe = {k: str(v)[:500] for k, v in flat.items()}
            self._mlflow.log_params(safe)
        self._flush()

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        flat = flatten_dict(metrics)
        numeric: dict[str, float] = {}
        for key, value in flat.items():
            try:
                numeric[key] = float(value)
            except (TypeError, ValueError):
                continue
        self._record["metrics"].update(numeric)
        if self._mlflow:
            self._mlflow.log_metrics(numeric, step=step)
        self._flush()

    def set_tags(self, tags: dict[str, Any]) -> None:
        clean = {k: str(v) for k, v in flatten_dict(tags).items() if v is not None}
        self._record["tags"].update(clean)
        if self._mlflow:
            self._mlflow.set_tags(clean)
        self._flush()

    def log_artifact(self, path: str | Path, *, copy_local: bool = False) -> None:
        p = Path(path)
        if not p.exists():
            return
        self._record["artifacts"].append(p.as_posix())
        if self._mlflow:
            if p.is_dir():
                self._mlflow.log_artifacts(str(p))
            else:
                self._mlflow.log_artifact(str(p))
        elif copy_local:
            dst = self.run_dir / "artifacts" / p.name
            if p.is_dir():
                shutil.copytree(p, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
        self._flush()

    def finish(self, status: str = "FINISHED", error: str | None = None) -> None:
        if not self._active:
            return
        self._record.update({"status": status, "ended_at": utc_now()})
        if error:
            self._record["error"] = error
        if self._mlflow:
            self._mlflow.end_run(status=status if status in {"FINISHED", "FAILED", "KILLED"} else "FINISHED")
        else:
            self._flush()
        self._active = False

    def _flush(self) -> None:
        if self.backend not in {"local", "mlflow"}:
            return
        # Even with MLflow, write a compact local mirror for Kaggle/offline lineage.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.run_dir / "run.json", self._record)

    def __enter__(self) -> "ExperimentTracker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish("FAILED" if exc else "FINISHED", repr(exc) if exc else None)
        return False
