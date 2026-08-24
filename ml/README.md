# Unified ML Layer

The maintained ML path is configuration-driven and model-agnostic:

- `data/` — frozen split, schema and leakage/fingerprint validation;
- `models/` — eight benchmark implementations behind one registry;
- `training/` — shared utilities, class balancing and resume checkpoints;
- `evaluation/` — one metric/calibration contract;
- `tuning/` — equal-budget Optuna search;
- `legacy_phobert/` — the previous single-model SentenAI trainer, retained only for provenance.

Use `python -m ml.train --model <name>`, not the files under `legacy_phobert/`.
