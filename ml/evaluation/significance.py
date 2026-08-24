from __future__ import annotations
import numpy as np
from .metrics import evaluate_predictions


def paired_bootstrap(y_true, pred_a, pred_b, *, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap confidence interval for the maintained primary F1 difference A-B."""
    y_true=np.asarray(y_true);pred_a=np.asarray(pred_a);pred_b=np.asarray(pred_b)
    if not (y_true.shape==pred_a.shape==pred_b.shape): raise ValueError("All arrays must share shape")
    rng=np.random.default_rng(seed);n=len(y_true);deltas=np.empty(n_boot,dtype=float)
    for i in range(n_boot):
        idx=rng.integers(0,n,n)
        a=evaluate_predictions(y_true[idx],pred_a[idx])["f1_combined"]
        b=evaluate_predictions(y_true[idx],pred_b[idx])["f1_combined"]
        deltas[i]=a-b
    observed=evaluate_predictions(y_true,pred_a)["f1_combined"]-evaluate_predictions(y_true,pred_b)["f1_combined"]
    lo,hi=np.quantile(deltas,[0.025,0.975]);p_two=2*min(float((deltas<=0).mean()),float((deltas>=0).mean()))
    return {"delta_f1_combined":round(float(observed),6),"ci95_low":round(float(lo),6),"ci95_high":round(float(hi),6),"p_two_sided":round(min(1.0,p_two),6),"n_boot":n_boot,"seed":seed}
