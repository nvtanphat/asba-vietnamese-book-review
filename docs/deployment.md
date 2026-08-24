# Deployment

`python scripts/deployment/promote_best.py` reads the benchmark leaderboard, filters to primary discriminative models and selects the highest **validation** `f1_combined`. The chosen experiment directory is copied to `artifacts/final/model`; calibrated thresholds are copied to `artifacts/final/thresholds.json`.

`absa_core.models.UnifiedArtifactPredictor` supports promoted classical, TextCNN, BiLSTM and pretrained-encoder artifacts and emits the existing SentenAI API contract. `apps/api/app/services/absa_service.py` prefers this unified artifact when `artifacts/final/metadata.json` exists; otherwise it falls back to the original Hugging Face PhoBERT predictor. This keeps a fresh clone usable before a new benchmark has been trained.

The legacy PhoBERT ONNX exporter remains under `packages/absa_core/scripts/export_onnx.py`. Unified ONNX export is deliberately gated until a winner has been promoted because graph signatures differ across model families. Do not silently export a different architecture under the legacy filename.
