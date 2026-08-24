# SentenAI Unified — Merge Report

## Scope completed

This repository merges the original `tiki-book-review-absa` research code and `sentenai-absa-dashboard` product code into one reproducible ABSA project.

### Research migration

- Included the provided `tiki-book-review_merged_fixed_v3.json` raw dataset.
- Migrated all 10 original Jupyter notebooks to auditable `.py` scripts under `scripts/migrated_notebooks/`.
- Moved the original specialized SentenAI PhoBERT trainer under `ml/legacy_phobert/` so it cannot be confused with the maintained benchmark path.
- Preserved the Tiki crawler and the data-quality Streamlit dashboard.

### Maintained fair benchmark

Eight registered models are available:

1. Logistic Regression
2. Linear SVM
3. TextCNN
4. BiLSTM
5. PhoBERT
6. XLM-RoBERTa-base
7. mDeBERTa-v3-base
8. ViT5 + LoRA (secondary generative benchmark)

The primary discriminative benchmark shares:

- one raw dataset;
- one semantic preprocessing implementation;
- one frozen split + SHA-256 manifest;
- one seed protocol;
- one 7-target schema;
- one evaluator;
- validation-only calibration, including calibrated validation scoring for neural early stopping;
- equal **completed-trial** Optuna search budget (default 20/model), with wall-clock time and parameter count reported separately;
- validation-only model selection.

Target formulation:

- overall sentiment: 3 classes (`negative`, `neutral`, `positive`);
- each of six aspects: 4 classes (`negative`, `neutral`, `positive`, `absent`).

Maintained selection metric:

`f1_combined = 0.5 * overall_sentiment_macro_f1 + 0.5 * mean(six aspect 4-class macro-F1)`

This formulation penalizes both missed aspects and hallucinated aspects.

### Production integration

- `packages/absa_core` is the shared preprocessing + inference source of truth.
- Fixed normalization-map resolution so training and serving use the same packaged emoji/vocabulary maps.
- `scripts/deployment/promote_best.py` promotes the validation winner to `artifacts/final/`.
- FastAPI prefers the promoted unified artifact.
- If no promoted artifact exists, API falls back to the original SentenAI PhoBERT/ONNX path.
- Existing Next.js dashboard, auth/database code and Docker deployment are retained.

## Dataset snapshot

Provided raw file:

- reviews: 13,412
- products: 2,009
- valid overall sentiment labels: 13,308
- aspect presence:
  - physical: 7,170
  - content: 5,276
  - delivery: 3,517
  - packaging: 3,206
  - service: 2,373
  - price: 970

The strong aspect/class imbalance is handled with macro metrics and bounded class weighting rather than accuracy-based model selection.

## Validation performed in the merge environment

- Maintained Python modules compile successfully.
- Project TOML and frontend JSON metadata parse successfully.
- 19 YAML configuration files parse successfully.
- Maintained deterministic unit/smoke suite: **8 passed**.
- Full API/package test collection requires runtime dependencies not installed in this offline sandbox (`slowapi`, `emoji`, `ftfy`, `pyvi`, `transformers`, `peft`); CI installs these from the declared project dependencies.
- Jupyter notebooks remaining in repository: **0**.
- Migrated notebook scripts: **10**.

## Deliberately not fabricated

A full deep-model benchmark was **not executed inside the merge sandbox**. The sandbox has no network access and does not currently contain all runtime/research dependencies or Hugging Face model weights (`slowapi`, `emoji`, `ftfy`, `pyvi`, `transformers`, `peft`, pretrained weights, etc.). The raw dataset and full pipeline are included, but generated train/validation/test rows and model leaderboard are intentionally not invented.

On a normal networked ML environment, run:

```bash
make install-ml
make prepare
make validate-data
python scripts/training/tune_all.py --trials 20
python -m ml.benchmark --use-tuned
python scripts/deployment/promote_best.py
```

The resulting leaderboard then becomes the authoritative comparison. Historical scores from old notebooks are provenance only and must not be mixed into it.

## Lockfile note

The pre-merge `uv.lock` was removed because it described the old workspace/package names and dependency graph. Run `uv lock` (or simply `uv sync`) once in a networked environment to generate a truthful lockfile for SentenAI Unified.

## Kaggle CLI orchestration added

The unified repo now includes `tools/kaggle_cli`, which shells out to the official Kaggle CLI rather than maintaining a second Kaggle API integration. It supports private dataset create/version, script-kernel packaging, explicit accelerator selection (T4 default), kernel status/log/output retrieval, validation-only remote training, and checkpoint resume by publishing `last.pt` to a small private Kaggle Dataset before the next kernel run. See `docs/kaggle_cli.md`.

## MLOps layer added

The merged repository now has an explicit MLOps control plane under `mlops/`:

- deterministic raw-data validation and SHA-256 lineage snapshots;
- DVC-ready data lifecycle (`dvc.yaml`, `params.yaml`, `bootstrap-dvc`);
- local JSON experiment tracking by default, with optional MLflow 3 tracking;
- tuning and training runs both record config, seed, test-seal policy, data/split fingerprints and metrics;
- automatic `MODEL_CARD.md`, `lineage.json`, and `run_manifest.json` per maintained model run;
- local immutable model-version registry with `candidate`, `staging`, and `champion` aliases;
- configurable quality gates before staging/production promotion;
- production promotion remains strictly validation-selected; test is a final report/gate and is no longer even a leaderboard tiebreaker;
- optional MLflow Model Registry bridge for the promoted unified artifact;
- FastAPI `/model-info` model traceability;
- privacy-preserving inference telemetry (no raw review text) with latency/model identity/text-shape features;
- input-drift reports based on Jensen-Shannon divergence;
- Kaggle `collect --register` flow to ingest completed remote runs into the same registry;
- MLOps GitHub Actions contract workflow and manual release-gate workflow;
- optional local MLflow server through the Docker Compose `mlops` profile.

The maintained deterministic unit/smoke suite is now **15 passed** in the merge sandbox after the MLOps additions.
