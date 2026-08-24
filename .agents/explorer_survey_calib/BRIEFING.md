# BRIEFING — 2026-08-24T00:28:40Z

## Mission
Explore and survey threshold calibration algorithms, absent bias, present-only macro F1 formulation, threshold storage/loading data flow, and evaluation pipeline in SentenAI-Unified.

## 🔒 My Identity
- Archetype: explorer
- Roles: Calibration Explorer, System Analyst
- Working directory: D:\vietcv\SentenAI-Unified\.agents\explorer_survey_calib
- Original parent: 95abf3c9-ecd1-4d03-9a74-a2869e75f3ee
- Milestone: Survey & Calibration Architecture Analysis

## 🔒 Key Constraints
- Read-only investigation — do NOT implement changes to production source code.
- Write reports, plans, and metadata only inside working directory `D:\vietcv\SentenAI-Unified\.agents\explorer_survey_calib`.
- Follow 5-Component Handoff Protocol (`handoff.md`).
- Communicate back to parent using `send_message`.

## Current Parent
- Conversation ID: 95abf3c9-ecd1-4d03-9a74-a2869e75f3ee
- Updated: 2026-08-24T00:28:40Z

## Investigation State
- **Explored paths**: `ml/evaluation/calibration.py`, `ml/evaluation/metrics.py`, `ml/evaluation/evaluator.py`, `ml/train.py`, `ml/benchmark.py`, `ml/models/transformer/model.py`, `ml/training/torch_text_trainer.py`, `packages/absa_core/absa_core/models/unified_predictor.py`, `experiments/*/metrics.json`.
- **Key findings**:
  1. Minority aspects (`as_price` 92.1% absent, `as_service` 82.0% absent) are severely penalized under 4-class macro F1 in `calibrate_absent_thresholds`.
  2. Because absent class has massive support, the grid search pushes thresholds to 0.90 to maximize absent accuracy, causing `f1_as_price` to crash to 0.136 and `f1_as_service` to crash to 0.148 in transformer models.
  3. Formulated and tested full-dataset 3-class present-only macro F1 (`labels=[0, 1, 2]`) with neutral protection, which penalizes both missed aspects and false-positive hallucinations while preventing absent label domination.
  4. Traced full threshold lifecycle: validation fitting -> in-epoch early stopping -> `evaluate_model` -> `metrics.json` -> promotion to `artifacts/final/thresholds.json` -> serving via `UnifiedArtifactPredictor`.
- **Unexplored areas**: None for survey scope.

## Key Decisions Made
- Selected Option B + Neutral Protection as the recommended objective formulation for `ml/evaluation/calibration.py`.

## Artifact Index
- `DISPATCH.md` — Dispatch log
- `BRIEFING.md` — Situational awareness
- `progress.md` — Liveness and progress tracking
- `handoff.md` — Complete 5-component survey report
