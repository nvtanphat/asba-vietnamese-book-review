"""Thresholds are calibrated automatically on validation by ml.train.
This utility prints the current per-aspect thresholds for a completed model."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--model",required=True);a=p.parse_args();payload=json.loads(Path(f"experiments/{a.model}/metrics.json").read_text(encoding="utf-8"));print(json.dumps(payload["thresholds"],indent=2))
