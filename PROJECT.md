# Project: SentenAI-Unified ABSA Optimization & Fair Benchmark Orchestration

## Architecture
- **Model Architecture & Modular Feature Pooling (`ml/models/transformer/`)**:
  - `ml/models/transformer/pooling.py`: Implements `FirstTokenPooling`, `MaskedMeanPooling`, `MultiHeadAttentionPooling`, and `build_pooling_layer`.
  - `ml/models/transformer/heads.py`: Implements `FlatMultiTaskHead`, `HierarchicalMultiTaskHead` (extracting overall sentiment latent $h_{\text{os}} \in \mathbb{R}^{128}$ to condition the 6 aspect classification heads), and `build_task_heads`.
  - `ml/models/transformer/model.py`: Unified `EncoderMultiTaskNetwork` accepting `pooling_type` and `head_type`, outputting standard `list[Tensor]` of length 7.
  - `packages/absa_core/absa_core/models/unified_architectures.py`: Synchronized `EncoderMultiTaskNetwork` to guarantee 100% production serving parity with `UnifiedArtifactPredictor`.
  - `ml/configs/models/phobert.yaml`, `mdeberta.yaml`, `xlmr.yaml`: Updated model hyperparameters (`pooling_type: masked_mean`, `head_type: hierarchical`).

- **Threshold Calibration Optimization (`ml/evaluation/calibration.py`)**:
  - `calibrate_absent_thresholds`: Implements full-dataset 3-class Present-Only Macro F1 objective (`labels=[0, 1, 2]`) with neutral protection weight ($w_{\text{neu}} = 0.15$), penalizing both false alarms and missed predictions without being dominated by the >92% absent class (`class 3`).
  - `decode_probabilities`: Decodes 7-task probabilities using calibrated presence thresholds $P(\text{present}) = 1 - P(\text{absent}) \ge t$.

- **Kaggle GPU Remote Orchestration (`tools/kaggle_cli/`)**:
  - `tools/kaggle_cli/cli.py`: Subcommands `doctor`, `prepare-data`, `sync-data`, `prepare-kernel`, `run`, `status`, `logs`, `output`, `resume`, `collect --register`.
  - Stages `sentenai_src_bundle.dat` and `data/splits/` to Kaggle Dataset.
  - Submits GPU jobs with `NvidiaTeslaT4`, executes validation-only (`--no-test`) during optimization and explicit unsealing (`--run-test`) for final evaluation.

- **Fair Benchmark Guardrails & MLOps (`docs/fair_benchmark.md`, `mlops/`)**:
  - Frozen dataset split: 70/15/15 stratified (`train`: 9300, `val`: 1991, `test`: 1992).
  - Stable lineage `data_fingerprint`: `c32f956aee64af890c0645d37da203a9ae1f62ca14d0ef85ac8f2182e3415623`.
  - Invariant random seed: `seed = 42`.
  - `ml/benchmark.py`: Generates `experiments/benchmark/leaderboard.csv`, updates `MODEL_CARD.md`, promotes champion model to `artifacts/final/`.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| 1 | Masked Mean & MHA Feature Pooling | Implement MaskedMeanPooling and MultiHeadAttentionPooling in ml/models/transformer/pooling.py | M1 | Survey Arch |
| 2 | Hierarchical Multi-Task Head | Condition 6 aspect sentiment branches on overall sentiment latent representation in ml/models/transformer/heads.py | M1 | Survey Arch |
| 3 | Transformer Model Integration | Integrate poolers and heads into EncoderMultiTaskNetwork and ml/configs/models/*.yaml | M1 | Survey Arch |
| 4 | Production Parity Sync | Synchronize EncoderMultiTaskNetwork in packages/absa_core/unified_architectures.py | M1 | Survey Arch |
| 5 | Present-Only Macro F1 Calibration | Implement 3-class Present-Only Macro F1 with neutral protection in ml/evaluation/calibration.py | M2 | Survey Calib |
| 6 | Minority Aspect Threshold Optimization | Ensure as_price and as_service thresholds avoid grid ceiling 0.90 and maintain F1 >= 0.40 | M2 | Survey Calib |
| 7 | Training & Validation Calibration Sync | Ensure transformer model validation & early stopping utilize new calibration objective | M2 | Survey Calib |
| 8 | Kaggle Dataset & Source Packaging | sync-data with sentenai_src_bundle.dat, data/splits, raw Tiki data | M3 | Survey Kaggle |
| 9 | Remote GPU Training Pipeline | Run remote Kaggle jobs (phobert, mdeberta, xlmr) with NvidiaTeslaT4 accelerator, monitor status/logs | M3 | Survey Kaggle |
| 10 | Artifact Sync & Registry Collection | Collect output artifacts to experiments/ and register candidate models | M3 | Survey Kaggle |
| 11 | Fair Benchmark Guardrails Verification | Verify data_fingerprint c32f956aee64af890c0645d37da203a9ae1f62ca14d0ef85ac8f2182e3415623, seed=42, test split sealing | M4 | Survey Kaggle |
| 12 | Final Evaluation (--run-test) & Leaderboard Update | Run final test evaluation, update experiments/benchmark/leaderboard.csv and MODEL_CARD.md | M4 | Survey Kaggle |
| 13 | Production Quality Gate & Promotion | Promote best model to artifacts/final/ ensuring test F1 combined > baseline (0.7280) and minority aspect F1 >= 0.40 | M4 | Survey Kaggle |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Transformer Architecture & Pooling Optimization | `ml/models/transformer/pooling.py`, `heads.py`, `model.py`, `packages/absa_core/absa_core/models/unified_architectures.py`, `ml/configs/models/*.yaml` | none | DONE |
| M2 | Calibration Algorithm Optimization for Minority Aspects | `ml/evaluation/calibration.py`, `tests/unit/test_calibration.py`, in-epoch validation alignment | M1 | DONE |
| M3 | Kaggle Remote GPU Training Execution & Collection | Dataset sync (`tools.kaggle_cli sync-data`), remote training of improved models (validation-only then final unsealed), status monitoring, artifact collection | M1, M2 | PLANNED |
| M4 | Fair Benchmark Evaluation, Leaderboard & Model Card Update | Verification of `data_fingerprint`, seed=42, `python -m ml.benchmark --promote-best`, `leaderboard.csv` update, `MODEL_CARD.md` generation, promotion | M3 | PLANNED |

---

## Interface Contracts

### 1. Feature Pooling Interface (`ml/models/transformer/pooling.py`)
- `forward(hidden_states: Tensor [B, L, D], attention_mask: Tensor [B, L]) -> Tensor [B, D]`
- Numerical stability clamp `sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)` for FP16 AMP.

### 2. Multi-Task Head Interface (`ml/models/transformer/heads.py`)
- `forward(pooled: Tensor [B, D]) -> list[Tensor]`
- Output tensor list length 7: index 0 is overall sentiment $[B, 3]$; indices 1..6 are aspect sentiments $[B, 4]$.

### 3. Model Output & Multitask Loss (`ml/training/losses.py`, `ml/models/transformer/model.py`)
- `EncoderMultiTaskNetwork.forward(input_ids, attention_mask)` returns `list[Tensor]` of length 7.
- `multitask_loss(logits: list[Tensor], labels: Tensor [B, 7], ...)` computes focal/cross-entropy loss.
- `predict_proba(texts: list[str]) -> list[np.ndarray]` returns list of 7 probability arrays (index 0: $(N, 3)$, indices 1..6: $(N, 4)$).

### 4. Calibration Interface (`ml/evaluation/calibration.py`)
- `calibrate_absent_thresholds(probabilities: list[np.ndarray], y_true: np.ndarray, grid=None, neutral_weight: float = 0.15) -> dict[str, float]`
- Returns mapping of aspect names (`as_content`, `as_physical`, `as_price`, `as_packaging`, `as_delivery`, `as_service`) to optimal float presence thresholds.

### 5. Kaggle CLI Interface (`tools/kaggle_cli/cli.py`)
- `python -m tools.kaggle_cli sync-data --dataset <owner/slug>`
- `python -m tools.kaggle_cli run --model <model> --accelerator NvidiaTeslaT4 [--no-test | --run-test]`
- `python -m tools.kaggle_cli status --kernel <owner/slug>`
- `python -m tools.kaggle_cli collect --model <model> --register`

---

## Code Layout
- `ml/models/transformer/pooling.py`: Feature pooling implementations (`MaskedMeanPooling`, `MultiHeadAttentionPooling`, `FirstTokenPooling`).
- `ml/models/transformer/heads.py`: Classification heads (`HierarchicalMultiTaskHead`, `FlatMultiTaskHead`).
- `ml/models/transformer/model.py`: Primary transformer model wrapper `EncoderMultiTaskNetwork`.
- `packages/absa_core/absa_core/models/unified_architectures.py`: Serving copy of `EncoderMultiTaskNetwork`.
- `ml/evaluation/calibration.py`: Threshold calibration and probability decoding.
- `ml/configs/models/*.yaml`: Model configuration files.
- `tools/kaggle_cli/`: Kaggle remote training orchestration tooling.
- `data/splits/`: Stratified frozen split JSONs and `split_manifest.json`.
- `experiments/`: Experiment runs, metrics, checkpoints, and benchmark leaderboard.
- `artifacts/final/`: Promoted production champion model and calibrated thresholds.
