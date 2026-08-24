# Original User Request

## 2026-08-23T15:50:03Z

This is a single self-contained fix; keep it small and focused.
Diagnose, harden and permanently resolve remote execution workflows for deep learning and transformer models (PhoBERT, mDeBERTa, XLM-R, ViT5) on Kaggle GPU infrastructure in SentenAI-Unified, ensuring robust dataset path resolution, package installation in ephemeral containers, project-scoped credential isolation, and automated artifact collection.

Working directory: D:/vietcv/SentenAI-Unified
Integrity mode: development

## Requirements

### R1. Robust Kaggle Remote Execution & Environment Bootstrap
The remote runner script and Kaggle CLI orchestration must dynamically locate the dataset and codebase regardless of nested /kaggle/input/ directory structures, automatically install internal packages (absa_core) and dependencies, and cleanly execute training runs under Kaggle GPU worker constraints.

### R2. Project-Scoped Credential Isolation
Kaggle authentication must strictly respect project-level configurations (.kaggle/kaggle.json / KAGGLE_CONFIG_DIR) without modifying or conflicting with system-wide user credentials, and ensure sensitive files remain ignored by version control.

### R3. Automated End-to-End Verification
Provide deterministic verification tests confirming that kernel generation, dataset staging, runner script compilation, and artifact collection function reliably.

## Acceptance Criteria

### Execution & Integration
- [ ] Runner script (run.py) recursively finds dataset files (train.json, val.json, test.json, sentenai_src.zip) inside any subfolder under /kaggle/input/.
- [ ] Local editable package absa_core installs without error in ephemeral remote environments.
- [ ] Output artifacts (best.pt, metrics.json, lineage.json) export to /kaggle/working/sentenai-output/ for automated collection.
- [ ] All unit and smoke tests in tests/unit/ pass 100%.
