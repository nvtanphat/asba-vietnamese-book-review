from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlops.data_quality import validate_dataset
from mlops.drift import build_reference_profile, check_drift
from mlops.gates import evaluate_quality_gate
from mlops.registry import LocalModelRegistry
from mlops.tracking import ExperimentTracker


def _dataset(path: Path):
    rows=[]
    for i in range(120):
        rows.append({
            "review_id": str(i), "content": "sách hay giao nhanh" if i % 2 else "sách hơi xấu",
            "sentiment": 2 if i % 2 else 0,
            "as_content": 2 if i % 2 else 0, "as_physical": None, "as_price": None,
            "as_packaging": None, "as_delivery": 2 if i % 2 else None, "as_service": None,
        })
    pd.DataFrame(rows).to_json(path, orient="records", force_ascii=False)


def test_data_quality_gate_and_tracking(tmp_path):
    data=tmp_path/"data.json"; _dataset(data)
    assert validate_dataset(data)["status"] == "PASS"
    gate=evaluate_quality_gate({"val_f1_combined":0.7,"test_f1_combined":0.67},{"min_val_f1_combined":0.5,"min_test_f1_combined":0.5,"max_generalization_gap":0.08,"require_test":True})
    assert gate["passed"]
    tracker=ExperimentTracker(experiment="test",root=tmp_path).start()
    tracker.log_params({"a":{"b":1}});tracker.log_metrics({"score":0.7});tracker.finish()
    payload=json.loads((tracker.run_dir/"run.json").read_text())
    assert payload["status"] == "FINISHED" and payload["params"]["a.b"] == 1


def test_registry_versions_and_aliases(tmp_path):
    artifact=tmp_path/"artifact";artifact.mkdir();(artifact/"metadata.json").write_text("{}")
    reg=LocalModelRegistry("registry.json",tmp_path)
    entry=reg.register(model="phobert",artifact_dir=artifact,metrics={"val_f1_combined":0.7,"test_f1_combined":0.68})
    out=reg.promote(model="phobert",version=entry["version"],alias="staging",gate={"min_val_f1_combined":0.5},release_root="releases")
    assert out["gate"]["passed"] and reg.resolve_alias("staging")["version"] == entry["version"]
    assert (tmp_path/"releases"/f"phobert-{entry['version']}"/"metadata.json").exists()


def test_drift_profile_detects_large_shift(tmp_path):
    ref=tmp_path/"ref.json";cur=tmp_path/"cur.json"
    pd.DataFrame({"text":["sách hay"]*120}).to_json(ref,orient="records")
    pd.DataFrame({"text":["A"*1000+" 123 !!!"]*120}).to_json(cur,orient="records")
    profile=build_reference_profile(ref)
    report=check_drift(profile,cur,warning_js=0.01,critical_js=0.05)
    assert report["status"] in {"WARNING","CRITICAL"}

def test_api_telemetry_never_stores_raw_text(tmp_path):
    from app.services.model_telemetry import record_prediction

    class Dummy:
        model_name = "phobert"
        family = "pretrained_encoder"

    secret = "đây là review riêng tư không được lưu nguyên văn"
    out = tmp_path / "telemetry.jsonl"
    record_prediction(
        text=secret,
        response={"overall":"positive","aspects":[{"aspect":"as_content"}]},
        predictor=Dummy(), path=out, latency_ms=12.3, sample_rate=1.0,
    )
    raw = out.read_text(encoding="utf-8")
    assert secret not in raw
    row = json.loads(raw)
    assert row["model"] == "phobert" and row["text_hash"] and row["text_length"] > 0
