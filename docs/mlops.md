# SentenAI MLOps

SentenAI keeps research fairness and production lifecycle separate but connected:

```text
raw data -> DVC/data snapshot -> frozen split -> train/tune on Kaggle or local
        -> experiment tracking -> validation-selected benchmark -> test once
        -> quality gate -> immutable model version -> staging/champion alias
        -> artifacts/final -> FastAPI -> privacy-preserving telemetry -> drift check
```

## 1. Install

Core MLOps commands use the normal workspace dependencies. MLflow and DVC are optional:

```bash
uv sync --group ml --group mlops
python -m mlops doctor
```

`SENTENAI_TRACKING_BACKEND=local` is the default and writes compact run metadata under `experiments/_tracking/`. To use MLflow:

```bash
export SENTENAI_TRACKING_BACKEND=mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow server --host 0.0.0.0 --port 5000
```

Training code does not change between backends.

## 2. Data lineage and DVC

Validate and fingerprint the exact dataset:

```bash
python -m mlops validate-data
python -m mlops snapshot-data
```

The snapshot records SHA-256, split-manifest hash when present, Git SHA, and environment metadata. Training writes the same data fingerprint into each model run.

To let DVC own the large raw dataset:

```bash
python -m mlops bootstrap-dvc
# then configure storage, for example:
dvc remote add -d storage <your-remote>
dvc push
```

`dvc.yaml` defines raw validation, frozen splitting, split validation, and generation of the drift reference profile.

## 3. Experiment tracking

Every `python -m ml.train --model ...` run logs:

- full flattened config;
- model/seed/test-seal tags;
- raw dataset + split fingerprints;
- validation/test metrics;
- training time and parameter count;
- `metrics.json`, `lineage.json`, split manifest and run manifest.

Kaggle runs use the same code, so their lineage is identical. After a remote run:

```bash
python -m tools.kaggle_cli collect \
  --owner USER --model phobert --register
```

This downloads the output into `experiments/phobert` and registers a candidate model version.

## 4. Registry and promotion

The local registry is `artifacts/registry/registry.json`. Versions are immutable release snapshots; aliases are mutable pointers:

- `candidate`: newest registered run;
- `staging`: passed staging gate;
- `champion`: production model.

Manual lifecycle:

```bash
python -m mlops register --model phobert --run-dir experiments/phobert
python -m mlops gate --run-dir experiments/phobert --stage production
python -m mlops promote --model phobert --version VERSION --stage production
```

The normal benchmark command can do the same automatically:

```bash
python -m ml.benchmark --promote-best
```

Selection remains **validation-only**. Test metrics are never used as a tiebreaker. Promotion then applies the production quality gate before assigning `champion` and updating `artifacts/final`.

Quality gates live in `mlops/config.yaml`, not in model code. Because the new fair benchmark has not yet been executed, the repository deliberately does **not** invent an absolute F1 floor. The initial production gate requires both validation/test metrics and limits their generalization gap; absolute minimum F1 thresholds should be added only after the first authoritative benchmark establishes a defensible baseline.

## 5. MLflow Model Registry (optional)

After a local champion exists:

```bash
python -m mlops mlflow-register \
  --artifact-dir artifacts/final \
  --name sentenai-absa \
  --alias champion
```

This wraps the unified artifact as an MLflow pyfunc model and registers a model version/alias. A remote tracking server is recommended for shared team use.

## 6. Production monitoring

FastAPI writes privacy-preserving telemetry when `ABSA_TELEMETRY_ENABLED=true` (default):

- UTC timestamp;
- short SHA-256 text hash, never raw text;
- text length, word count, ASCII/digit/punctuation ratios;
- predicted overall/aspects;
- model/family;
- inference latency.

Configuration:

```env
ABSA_TELEMETRY_ENABLED=true
ABSA_TELEMETRY_PATH=data/model_telemetry.jsonl
ABSA_TELEMETRY_SAMPLE_RATE=1.0
```

Create a training reference and compare live traffic:

```bash
python -m mlops profile-data --input data/splits/train.json
python -m mlops drift --current apps/api/data/model_telemetry.jsonl --fail-on-critical
```

Drift uses Jensen-Shannon divergence on stable, label-free text features. `mlops/config.yaml` contains warning/critical thresholds. This detects input drift; it does **not** replace ground-truth performance monitoring once delayed labels become available.

The API exposes `/model-info` so deployments can be traced to registry version and data fingerprint without loading model weights.

## 7. CI/CD gates

`.github/workflows/mlops.yml` validates raw-data schema, verifies the committed dataset fingerprint, compiles MLOps/train/Kaggle code, and runs contract tests on every PR. `.github/workflows/model-release.yml` provides a manual, gated model release check before building the API image.

## 8. Recommended team flow

1. Change code/data on a branch.
2. CI validates contracts and data fingerprint.
3. Run equal-budget tuning/training on Kaggle with test sealed.
4. Select on validation only.
5. Run test once for the frozen candidate.
6. Register model version.
7. Production quality gate.
8. Promote to `champion` / `artifacts/final`.
9. Deploy API image.
10. Monitor latency/input drift; retrain only when there is evidence for it.
