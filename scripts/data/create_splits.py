import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""Canonical preprocessing entry point: frozen split + shared text cleaning."""
from ml.data.split import main
if __name__ == "__main__": main()
