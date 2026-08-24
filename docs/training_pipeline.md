# Training Pipeline

```text
raw v3 JSON
   ↓
shared preprocessing
   ↓
frozen 70/15/15 split + manifest
   ↓
train only ──────────────┐
                         ├─ model fitting
validation ──────────────┤  calibrated early stopping + threshold calibration + tuning
                         ↓
                validation leaderboard
                         ↓
        choose winner by validation F1
                         ↓
                  sealed test report
                         ↓
              artifacts/final/model
                         ↓
                   FastAPI → Next.js
```

## Commands

```bash
make install-ml
make prepare
make validate-data

# one model
python -m ml.train --model linear_svm
python -m ml.train --model phobert --resume

# equal-budget tuning without test selection
python -m ml.tune --model phobert --trials 20
python -m ml.train --model phobert --use-tuned

# equal completed-trial budget for all models
python scripts/training/tune_all.py --trials 20

# all benchmark models using tuned configs
make benchmark-tuned
python scripts/deployment/promote_best.py
```

Neural and transformer trainers save `last.pt` so interrupted jobs can resume with `--resume`. Best checkpoints are selected only from validation score.
