import pandas as pd
from pathlib import Path

def test_raw_dataset_snapshot():
    root=Path(__file__).resolve().parents[2]
    df=pd.read_json(root/'data/raw/tiki-book-review_merged_fixed_v3.json')
    assert len(df)==13412
    assert df['product_id'].nunique()==2009
    assert df['review_id'].duplicated().sum()==0
