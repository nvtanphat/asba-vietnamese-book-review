from pathlib import Path
import pandas as pd
p=Path("experiments/benchmark/leaderboard.csv")
if not p.exists(): raise SystemExit("Run benchmark first")
df=pd.read_csv(p).sort_values("val_f1_combined",ascending=False);print(df.to_string(index=False))
