from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from ml.data import TARGET_COLS, load_splits
from ml.evaluation.evaluator import evaluate_model
from ml.evaluation.benchmark import update_leaderboard
from ml.models import build_model
from ml.utils import deep_merge, load_yaml, seed_everything
from mlops.lineage import dataset_snapshot
from mlops.tracking import ExperimentTracker
from mlops.model_card import generate_model_card

ROOT = Path(__file__).resolve().parents[1]


def load_config(model_name: str, use_tuned: bool = False) -> dict:
    base = load_yaml(ROOT / "ml/configs/base.yaml")
    model_cfg = load_yaml(ROOT / f"ml/configs/models/{model_name}.yaml")
    cfg = deep_merge(base, model_cfg)
    if use_tuned:
        tuned_dir = ROOT / f"experiments/tuning/{model_name}"
        tuned = tuned_dir / "best_config.yaml"
        manifest = tuned_dir / "tuning_manifest.json"
        required = int(base.get("fairness", {}).get("tuning_trials_per_model", 20))
        if not tuned.exists() or not manifest.exists():
            raise FileNotFoundError(f"{model_name} has no completed tuned config. Run: python -m ml.tune --model {model_name} --trials {required}")
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if int(info.get("completed_trials", 0)) < required:
            raise RuntimeError(f"{model_name} tuning is incomplete: {info.get('completed_trials', 0)}/{required} completed trials")
        cfg = deep_merge(cfg, load_yaml(tuned))
    cfg["name"] = model_name
    return cfg


def ensure_splits(data_root: Path):
    needed = [data_root / f"{s}.json" for s in ["train", "val", "test"]]
    if not all(p.exists() for p in needed):
        from ml.data.split import create_splits
        create_splits(ROOT / "data/raw/tiki-book-review_merged_fixed_v3.json", data_root)


def run(model_name: str, *, resume: bool = False, use_tuned: bool = False, run_test: bool = True):
    cfg = load_config(model_name, use_tuned)
    seed = int(cfg.get("seed", 42)); seed_everything(seed)
    data_root = ROOT / cfg.get("data_root", "data/splits"); ensure_splits(data_root)
    train_df, val_df, test_df = load_splits(data_root)
    y_train = train_df[TARGET_COLS].to_numpy(dtype=int)
    y_val = val_df[TARGET_COLS].to_numpy(dtype=int)
    y_test = test_df[TARGET_COLS].to_numpy(dtype=int)

    run_dir = ROOT / "experiments" / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    lineage = dataset_snapshot(
        ROOT / "data/raw/tiki-book-review_merged_fixed_v3.json",
        root=ROOT,
        split_manifest=data_root / "split_manifest.json",
        include_environment=True,
    )
    tracker = ExperimentTracker(
        experiment="sentenai-absa",
        run_name=f"{model_name}-seed{seed}",
        root=ROOT,
    ).start()
    tracker.log_params({
        "model": model_name,
        "seed": seed,
        "resume": resume,
        "use_tuned": use_tuned,
        "run_test": run_test,
        "config": cfg,
    })
    tracker.set_tags({
        "model": model_name,
        "data_fingerprint": lineage.get("fingerprint"),
        "split_manifest_sha256": lineage.get("split_manifest", {}).get("sha256"),
        "git_sha": lineage.get("git_sha"),
        "test_policy": "unsealed" if run_test else "sealed",
    })

    try:
        model = build_model(model_name, cfg)
        started = time.perf_counter()
        model.fit(train_df["text"].tolist(), y_train, val_df["text"].tolist(), y_val, output_dir=run_dir, resume=resume)
        train_seconds = time.perf_counter() - started

        if run_test:
            results = evaluate_model(model, val_df["text"].tolist(), y_val, test_df["text"].tolist(), y_test, output_dir=run_dir)
        else:
            from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
            from ml.evaluation.metrics import evaluate_predictions
            val_probs = model.predict_proba(val_df["text"].tolist())
            thresholds = calibrate_absent_thresholds(val_probs, y_val)
            results = {"thresholds": thresholds, "val": evaluate_predictions(y_val, decode_probabilities(val_probs, thresholds)), "test": {}}
            (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

        row = {
            "model": model_name,
            "family": getattr(model, "family", "unknown"),
            "val_f1_combined": results["val"]["f1_combined"],
            "test_f1_combined": results["test"].get("f1_combined", np.nan),
            "test_f1_sentiment": results["test"].get("f1_sentiment", np.nan),
            "test_f1_aspect_presence": results["test"].get("f1_aspect_presence", np.nan),
            "test_f1_aspect_present": results["test"].get("f1_aspect_present", np.nan),
            "test_f1_aspect_4class_mean": results["test"].get("f1_aspect_4class_mean", np.nan),
            "params": model.parameter_count(),
            "train_seconds": round(train_seconds, 2),
            "seed": seed,
            "tracking_run_id": tracker.run_id,
            "data_fingerprint": lineage.get("fingerprint"),
        }
        tracker.log_metrics({
            "val": results.get("val", {}),
            "test": results.get("test", {}),
            "train_seconds": train_seconds,
            "params": model.parameter_count(),
        })
        tracker.log_artifact(run_dir / "metrics.json")
        tracker.log_artifact(data_root / "split_manifest.json")
        (run_dir / "lineage.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
        tracker.log_artifact(run_dir / "lineage.json")
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "model": model_name,
                "tracking_run_id": tracker.run_id,
                "data_fingerprint": lineage.get("fingerprint"),
                "test_policy": "unsealed" if run_test else "sealed",
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tracker.log_artifact(run_dir / "run_manifest.json")
        generate_model_card(
            model=model_name,
            metrics={
                "val_f1_combined": row["val_f1_combined"],
                "test_f1_combined": row["test_f1_combined"],
                "test_f1_sentiment": row["test_f1_sentiment"],
                "test_f1_aspect_4class_mean": row["test_f1_aspect_4class_mean"],
            },
            lineage=lineage,
            output=run_dir / "MODEL_CARD.md",
            notes="Generated automatically by the unified training pipeline.",
        )
        tracker.log_artifact(run_dir / "MODEL_CARD.md")
        if run_test: update_leaderboard(row, ROOT / "experiments/benchmark")
        tracker.finish("FINISHED")
        print(json.dumps({"row": row, "results": results}, ensure_ascii=False, indent=2))
        return row, results
    except Exception as exc:
        tracker.finish("FAILED", repr(exc))
        raise


def main():
    p = argparse.ArgumentParser(description="Train one model under the frozen SentenAI fair-benchmark protocol.")
    p.add_argument("--model", required=True, choices=["logistic","linear_svm","textcnn","bilstm","phobert","xlmr","mdeberta","vit5"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--use-tuned", action="store_true")
    p.add_argument("--no-test", action="store_true", help="Validation only; use this for tuning to keep test sealed.")
    args = p.parse_args()
    run(args.model, resume=args.resume, use_tuned=args.use_tuned, run_test=not args.no_test)


if __name__ == "__main__": main()
