from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .common import atomic_write_json, read_json
from .data_quality import validate_dataset
from .drift import build_reference_profile, check_drift
from .gates import evaluate_quality_gate
from .lineage import dataset_snapshot, write_dataset_snapshot
from .mlflow_bridge import register_unified_artifact
from .model_card import generate_model_card
from .registry import LocalModelRegistry

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "mlops/config.yaml"


def cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}


def _metrics_from_run(run_dir: Path) -> dict:
    p = run_dir / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    row = {
        "val_f1_combined": payload.get("val", {}).get("f1_combined"),
        "test_f1_combined": payload.get("test", {}).get("f1_combined"),
        "test_f1_sentiment": payload.get("test", {}).get("f1_sentiment"),
        "test_f1_aspect_4class_mean": payload.get("test", {}).get("f1_aspect_4class_mean"),
    }
    return {k: v for k, v in row.items() if v is not None}


def cmd_doctor(_args):
    optional = {}
    for name in ("mlflow", "dvc"):
        try:
            mod = __import__(name)
            optional[name] = getattr(mod, "__version__", "installed")
        except Exception:
            optional[name] = "not installed (optional)"
    print(json.dumps({"python": sys.version.split()[0], "tracking_backend": os.getenv("SENTENAI_TRACKING_BACKEND", "local"), "optional": optional}, indent=2))


def cmd_validate_data(args):
    result = validate_dataset(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


def cmd_snapshot(args):
    c = cfg(); out = Path(args.output or c["lineage"]["snapshot_path"])
    payload = dataset_snapshot(args.input, root=ROOT, split_manifest=args.split_manifest)
    if args.check and out.exists():
        previous = read_json(out, {})
        if previous.get("fingerprint") != payload.get("fingerprint"):
            print(json.dumps({"status":"FAIL","expected":previous.get("fingerprint"),"actual":payload.get("fingerprint")}, indent=2))
            raise SystemExit(2)
        print(json.dumps({"status":"PASS","fingerprint":payload["fingerprint"]}, indent=2))
        return
    atomic_write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_profile(args):
    c = cfg(); out = args.output or c["monitoring"]["reference_profile"]
    profile = build_reference_profile(args.input, ROOT / out if not Path(out).is_absolute() else out)
    print(json.dumps({"status":"PASS","count":profile["count"],"output":str(out)}, indent=2))


def cmd_drift(args):
    c = cfg(); m = c["monitoring"]
    report = check_drift(args.reference, args.current, warning_js=float(m["warning_js"]), critical_js=float(m["critical_js"]))
    out = Path(args.output or m["drift_report"])
    if not out.is_absolute(): out = ROOT / out
    atomic_write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["current_count"] < int(m.get("min_samples", 100)):
        print(f"warning: only {report['current_count']} samples; configured minimum is {m.get('min_samples',100)}", file=sys.stderr)
    if args.fail_on_critical and report["status"] == "CRITICAL":
        raise SystemExit(3)


def cmd_gate(args):
    c = cfg(); metrics = _metrics_from_run(Path(args.run_dir)); gate = c["quality_gates"][args.stage]
    result = evaluate_quality_gate(metrics, gate)
    print(json.dumps({"stage":args.stage,"metrics":metrics,"gate":result}, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def cmd_register(args):
    c = cfg(); r_cfg = c["registry"]
    run_dir = Path(args.run_dir)
    metrics = _metrics_from_run(run_dir)
    snap_path = ROOT / c["lineage"]["snapshot_path"]
    lineage = read_json(snap_path, {})
    registry = LocalModelRegistry(r_cfg["path"], ROOT)
    entry = registry.register(model=args.model, artifact_dir=run_dir, metrics=metrics, lineage=lineage, tracking_run_id=args.tracking_run_id, notes=args.notes)
    print(json.dumps(entry, ensure_ascii=False, indent=2))


def cmd_promote(args):
    c = cfg(); r_cfg = c["registry"]
    registry = LocalModelRegistry(r_cfg["path"], ROOT)
    alias = {"candidate":"candidate", "staging":"staging", "production":"champion"}[args.stage]
    result = registry.promote(
        model=args.model,
        version=args.version,
        alias=alias,
        gate=c["quality_gates"][args.stage],
        release_root=r_cfg["release_root"],
        production_artifact=r_cfg["production_artifact"] if args.stage == "production" else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_card(args):
    c = cfg(); run_dir=Path(args.run_dir); metrics=_metrics_from_run(run_dir); lineage=read_json(ROOT/c["lineage"]["snapshot_path"],{})
    out = Path(args.output or (run_dir / "MODEL_CARD.md"))
    generate_model_card(model=args.model, metrics=metrics, lineage=lineage, output=out, notes=args.notes or "")
    print(out)


def cmd_mlflow(args):
    c=cfg(); r=c["registry"]
    result=register_unified_artifact(args.artifact_dir,registered_model_name=args.name or r["registered_model_name"],alias=args.alias,tracking_uri=args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_bootstrap_dvc(args):
    exe=shutil.which("dvc")
    if not exe:
        raise SystemExit("dvc is not installed. Run: uv sync --group mlops")
    if not (ROOT/".dvc").exists(): subprocess.run([exe,"init"],cwd=ROOT,check=True)
    data=Path(args.input)
    subprocess.run([exe,"add",str(data)],cwd=ROOT,check=True)
    print("DVC initialized. Configure a remote with `dvc remote add -d <name> <url>` before pushing data.")


def build_parser():
    p=argparse.ArgumentParser(prog="python -m mlops",description="SentenAI MLOps control plane")
    s=p.add_subparsers(dest="command",required=True)
    q=s.add_parser("doctor"); q.set_defaults(func=cmd_doctor)
    q=s.add_parser("validate-data"); q.add_argument("--input",default="data/raw/tiki-book-review_merged_fixed_v3.json"); q.set_defaults(func=cmd_validate_data)
    q=s.add_parser("snapshot-data"); q.add_argument("--input",default="data/raw/tiki-book-review_merged_fixed_v3.json");q.add_argument("--output");q.add_argument("--split-manifest",default="data/splits/split_manifest.json");q.add_argument("--check",action="store_true");q.set_defaults(func=cmd_snapshot)
    q=s.add_parser("profile-data");q.add_argument("--input",default="data/splits/train.json");q.add_argument("--output");q.set_defaults(func=cmd_profile)
    q=s.add_parser("drift");q.add_argument("--reference",default="artifacts/monitoring/reference_profile.json");q.add_argument("--current",required=True);q.add_argument("--output");q.add_argument("--fail-on-critical",action="store_true");q.set_defaults(func=cmd_drift)
    q=s.add_parser("gate");q.add_argument("--run-dir",required=True);q.add_argument("--stage",choices=["candidate","staging","production"],required=True);q.set_defaults(func=cmd_gate)
    q=s.add_parser("register");q.add_argument("--model",required=True);q.add_argument("--run-dir",required=True);q.add_argument("--tracking-run-id");q.add_argument("--notes");q.set_defaults(func=cmd_register)
    q=s.add_parser("promote");q.add_argument("--model",required=True);q.add_argument("--version",required=True);q.add_argument("--stage",choices=["candidate","staging","production"],required=True);q.set_defaults(func=cmd_promote)
    q=s.add_parser("model-card");q.add_argument("--model",required=True);q.add_argument("--run-dir",required=True);q.add_argument("--output");q.add_argument("--notes");q.set_defaults(func=cmd_card)
    q=s.add_parser("mlflow-register");q.add_argument("--artifact-dir",default="artifacts/final");q.add_argument("--name");q.add_argument("--alias",default="champion");q.add_argument("--tracking-uri");q.set_defaults(func=cmd_mlflow)
    q=s.add_parser("bootstrap-dvc");q.add_argument("--input",default="data/raw/tiki-book-review_merged_fixed_v3.json");q.set_defaults(func=cmd_bootstrap_dvc)
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    args.func(args)

if __name__=="__main__": main()
