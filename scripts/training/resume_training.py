import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from ml.train import run
p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--use-tuned",action="store_true");a=p.parse_args();run(a.model,resume=True,use_tuned=a.use_tuned,run_test=True)
