from pathlib import Path
import pandas as pd
p=Path("experiments/benchmark/leaderboard.csv");out=Path("artifacts/reports/tables/leaderboard.md");out.parent.mkdir(parents=True,exist_ok=True)
if not p.exists(): raise SystemExit("Run benchmark first")
df=pd.read_csv(p).sort_values("val_f1_combined",ascending=False);out.write_text("# Fair Benchmark Leaderboard\n\n"+df.to_markdown(index=False)+"\n",encoding="utf-8");print(out)
