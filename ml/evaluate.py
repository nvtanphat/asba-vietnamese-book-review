"""Metrics are produced automatically by ml.train/ml.benchmark.
This entry point prints the current frozen leaderboard without retraining."""
from pathlib import Path
import pandas as pd

p=Path("experiments/benchmark/leaderboard.csv")
if not p.exists(): raise SystemExit("No leaderboard yet. Run: python -m ml.benchmark")
print(pd.read_csv(p).to_string(index=False))
