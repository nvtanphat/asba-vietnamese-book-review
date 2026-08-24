from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PRIMARY_COLUMNS = [
    "model", "family", "val_f1_combined", "test_f1_combined", "test_f1_sentiment",
    "test_f1_aspect_presence", "test_f1_aspect_present", "params", "train_seconds", "seed"
]


def update_leaderboard(row: dict, output_dir: str | Path = "experiments/benchmark") -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "leaderboard.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["model"] != row["model"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values(["val_f1_combined", "model"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    (out / "leaderboard.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    if len(df):
        best = df.iloc[0].to_dict()
        (out / "best_model.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return df
