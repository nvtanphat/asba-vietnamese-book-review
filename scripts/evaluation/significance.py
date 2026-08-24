from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse,json,numpy as np
from ml.data import TARGET_COLS,load_splits
from ml.evaluation.significance import paired_bootstrap
p=argparse.ArgumentParser();p.add_argument("model_a");p.add_argument("model_b");p.add_argument("--n-boot",type=int,default=2000);a=p.parse_args();_,_,test=load_splits("data/splits");y=test[TARGET_COLS].to_numpy(int);pa=np.load(f"experiments/{a.model_a}/test_predictions.npy");pb=np.load(f"experiments/{a.model_b}/test_predictions.npy");print(json.dumps(paired_bootstrap(y,pa,pb,n_boot=a.n_boot),indent=2))
