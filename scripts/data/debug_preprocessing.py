import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""Maintained CLI replacement for notebook 00_preprocessing_debug.ipynb."""
import argparse, pandas as pd
from absa_core.preprocessing.pipeline import clean_text_series
p=argparse.ArgumentParser();p.add_argument("--input",default="data/raw/tiki-book-review_merged_fixed_v3.json");p.add_argument("--n",type=int,default=20);a=p.parse_args();df=pd.read_json(a.input).head(a.n);df["cleaned"]=clean_text_series(df["content"]);print(df[["content","cleaned"]].to_string(index=False))
