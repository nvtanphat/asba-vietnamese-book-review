from __future__ import annotations
import pandas as pd
from ml.data.schema import ASPECT_COLS

def dataset_summary(df: pd.DataFrame) -> dict:
    return {"rows":len(df),"products":int(df.product_id.nunique()) if "product_id" in df else None,"sentiment":df.sentiment.value_counts(dropna=False).to_dict(),"aspect_presence":{c:int(df[c].notna().sum()) for c in ASPECT_COLS}}
