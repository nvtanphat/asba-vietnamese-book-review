from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd

from ml.train import run, ROOT
from ml.utils.io import load_yaml
from mlops.common import read_json
from mlops.registry import LocalModelRegistry


def promote_best(primary_models: list[str]):
    board_path = ROOT / "experiments/benchmark/leaderboard.csv"
    if not board_path.exists(): raise FileNotFoundError("No leaderboard found. Run benchmark first.")
    board = pd.read_csv(board_path)
    # Selection is validation-only. Test metrics are reported, never used as a tiebreaker.
    eligible = board[board["model"].isin(primary_models)].sort_values(["val_f1_combined", "model"], ascending=[False, True])
    if eligible.empty: raise RuntimeError("No primary benchmark models were completed.")
    best = eligible.iloc[0].to_dict(); name = best["model"]
    src = ROOT / "experiments" / name; dst = ROOT / "artifacts/final/model"
    if dst.exists(): shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)

    lineage = read_json(src / "lineage.json", {})
    metadata = {
        "selected_by": "validation f1_combined",
        "model": name,
        "leaderboard_row": best,
        "data_fingerprint": lineage.get("fingerprint"),
        "git_sha": lineage.get("git_sha"),
    }
    (ROOT / "artifacts/final/metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if (src / "metrics.json").exists():
        payload=json.loads((src/"metrics.json").read_text(encoding="utf-8"))
        (ROOT/"artifacts/final/thresholds.json").write_text(json.dumps(payload.get("thresholds",{}),ensure_ascii=False,indent=2),encoding="utf-8")

    # Register immutable release + run the production quality gate before assigning champion.
    mlops_cfg = load_yaml(ROOT / "mlops/config.yaml")
    r_cfg = mlops_cfg["registry"]
    registry = LocalModelRegistry(r_cfg["path"], ROOT)
    entry = registry.register(
        model=name,
        artifact_dir=ROOT / "artifacts/final",
        metrics=best,
        lineage=lineage,
        tracking_run_id=str(best.get("tracking_run_id")) if best.get("tracking_run_id") else None,
        notes="Selected strictly by validation f1_combined from fair benchmark.",
    )
    promotion = registry.promote(
        model=name,
        version=entry["version"],
        alias="champion",
        gate=mlops_cfg["quality_gates"]["production"],
        release_root=r_cfg["release_root"],
        production_artifact=r_cfg["production_artifact"],
    )
    meta_path = ROOT / "artifacts/final/metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["registry_version"] = entry["version"]
    metadata["registry_alias"] = "champion"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Promoted {name}@{entry['version']} -> champion -> artifacts/final (validation-selected, production-gated)")
    return name


def main():
    p=argparse.ArgumentParser();p.add_argument("--models",nargs="*");p.add_argument("--resume",action="store_true");p.add_argument("--use-tuned",action="store_true");p.add_argument("--promote-best",action="store_true");args=p.parse_args()
    cfg=load_yaml(ROOT/"ml/configs/benchmark.yaml");models=args.models or cfg["models"]
    for name in models:
        print(f"\n===== {name} =====")
        run(name,resume=args.resume,use_tuned=args.use_tuned,run_test=True)
    if args.promote_best: promote_best(cfg["primary_models"])


if __name__=="__main__":main()
