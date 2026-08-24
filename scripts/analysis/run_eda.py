from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse,json
from pathlib import Path
import pandas as pd
from ml.data.schema import ASPECT_COLS
p=argparse.ArgumentParser();p.add_argument("--input",default="data/raw/tiki-book-review_merged_fixed_v3.json");p.add_argument("--output",default="artifacts/reports/data_summary.json");a=p.parse_args();df=pd.read_json(a.input)
summary={"rows":len(df),"products":int(df.product_id.nunique()) if "product_id" in df else None,"labeled_sentiment":int(df.sentiment.notna().sum()),"missing_content":int(df.content.isna().sum()),"sentiment_distribution":df.sentiment.value_counts(dropna=False).sort_index().to_dict(),"aspect_presence":{c:{"count":int(df[c].notna().sum()),"rate":round(float(df[c].notna().mean()),6)} for c in ASPECT_COLS}}
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
