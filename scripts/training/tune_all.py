from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from ml.models.registry import MODEL_REGISTRY
from ml.train import load_config
from ml.tuning.tuner import tune


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune every selected SentenAI benchmark model with the same completed-trial budget."
    )
    parser.add_argument("--models", nargs="*", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--trials", type=int, default=None, help="Completed Optuna trials required per model.")
    args = parser.parse_args()

    protocol_budget = int(load_config("logistic").get("fairness", {}).get("tuning_trials_per_model", 20))
    budget = args.trials if args.trials is not None else protocol_budget
    if budget < 1:
        raise ValueError("--trials must be >= 1")

    models = args.models or list(MODEL_REGISTRY)
    for name in models:
        print(f"===== tuning {name}: {budget} completed trials =====")
        tune(name, budget)


if __name__ == "__main__":
    main()
