import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.benchmark import promote_best, ROOT
from ml.utils.io import load_yaml
cfg=load_yaml(ROOT/"ml/configs/benchmark.yaml");promote_best(cfg["primary_models"])
