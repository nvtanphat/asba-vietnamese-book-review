from __future__ import annotations
import argparse, json
from pathlib import Path
from absa_core.models.unified_predictor import UnifiedArtifactPredictor

p=argparse.ArgumentParser();p.add_argument("text");p.add_argument("--artifact-dir",default="artifacts/final")
a=p.parse_args();print(json.dumps(UnifiedArtifactPredictor(a.artifact_dir).predict(a.text)[0],ensure_ascii=False,indent=2))
