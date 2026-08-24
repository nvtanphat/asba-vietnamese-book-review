from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""Leakage-safe 5-fold diagnostic for classical baselines only.
The frozen train/val/test benchmark remains the source of truth; CV never touches test.
"""
import argparse, numpy as np
from sklearn.model_selection import StratifiedKFold
from ml.data import TARGET_COLS, load_splits
from ml.models import build_model
from ml.train import load_config
from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
from ml.evaluation.metrics import evaluate_predictions
p=argparse.ArgumentParser();p.add_argument("--model",choices=["logistic","linear_svm"],default="logistic");a=p.parse_args();train,val,_=load_splits("data/splits");df=__import__('pandas').concat([train,val],ignore_index=True);y=df[TARGET_COLS].to_numpy(int);skf=StratifiedKFold(5,shuffle=True,random_state=42);scores=[]
for fold,(tr,va) in enumerate(skf.split(df,df.sentiment),1):
 m=build_model(a.model,load_config(a.model));m.fit(df.text.iloc[tr].tolist(),y[tr]);pva=m.predict_proba(df.text.iloc[va].tolist());th=calibrate_absent_thresholds(pva,y[va]);pred=decode_probabilities(pva,th);score=evaluate_predictions(y[va],pred)["f1_combined"];scores.append(score);print(f"fold={fold} f1_combined={score:.4f}")
print(f"mean={np.mean(scores):.4f} std={np.std(scores):.4f}")
