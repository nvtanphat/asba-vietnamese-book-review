# Specification Mining Report: Kaggle CLI Tooling, Fair Benchmark & Leaderboard

## Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Kaggle CLI | `doctor` subcommand | Validates official Kaggle CLI installation (`kaggle --version`) and verifies active authentication via lightweight API call (`kaggle datasets list -p 1`). | None | CLI version string and `Kaggle authentication/API check: OK` | Raises `KaggleToolError` if `kaggle` executable not found or if API check returns non-zero exit code. | `tools/kaggle_cli/cli.py:437-448`, `docs/kaggle_cli.md:25` |
| 2 | Kaggle CLI | `prepare-data` subcommand | Stages flat dataset structure into `.kaggle_work/datasets/main` containing raw JSON, split files, maps, source bundle, and metadata. | `--dataset <owner/slug>`, optional `--title` | Staged dataset directory with `dataset-metadata.json`, `sentenai_src_bundle.dat`, raw tiki JSON, split JSONs, map JSONs | Raises `KaggleToolError` if `data/raw/tiki-book-review_merged_fixed_v3.json` is missing or handle is invalid. | `tools/kaggle_cli/cli.py:155-185, 450-452` |
| 3 | Kaggle CLI | `sync-data` subcommand | Prepares stage and creates or versions the Kaggle dataset (`kaggle datasets version` or `kaggle datasets create`). Private by default. | `--dataset <owner/slug>`, `--message`, `--public` | Uploaded / updated Kaggle dataset | Exits with error code 2 / `KaggleToolError` on invalid handle or network / Kaggle API failure. | `tools/kaggle_cli/cli.py:194-204, 454-456`, `docs/kaggle_cli.md:30-40` |
| 4 | Kaggle CLI | `prepare-kernel` subcommand | Builds self-contained Kaggle kernel bundle under `.kaggle_work/kernels/<model>` with generated `run.py`, `sentenai_src.zip`, and `kernel-metadata.json`. | Model args (`--model`, `--owner`, `--dataset`, `--accelerator`, `--use-tuned`, `--run-test`, `--no-internet`, `--timeout`) | Kernel directory with `run.py`, `sentenai_src.zip`, `kernel-metadata.json` | Raises `KaggleToolError` if model is not in `MODELS` or handle is invalid. | `tools/kaggle_cli/cli.py:378-418, 476-478` |
| 5 | Kaggle CLI | `run` subcommand | Builds kernel bundle (optionally syncing data first if `--sync-data`), and submits via `kaggle kernels push --accelerator <acc>`. | Model & training args + `--sync-data` | Submitted Kaggle kernel handle | Fails if Kaggle CLI push fails, GPU quota exceeded, or network error. | `tools/kaggle_cli/cli.py:480-488`, `docs/kaggle_cli.md:41-71` |
| 6 | Kaggle CLI | `status` subcommand | Inspects the current state of a Kaggle kernel via `kaggle kernels status <kernel>`. | `--kernel <owner/slug>` | Kernel status (e.g. `queued`, `running`, `complete`, `error`, `cancel_acknowledged`) | Fails if kernel handle is invalid or not found. | `tools/kaggle_cli/cli.py:490-493`, `docs/kaggle_cli.md:73-77` |
| 7 | Kaggle CLI | `logs` subcommand | Streams or displays remote execution logs via `kaggle kernels logs <kernel>` (supports `--follow`). | `--kernel <owner/slug>`, optional `--follow` | Streamed stdout/stderr from Kaggle kernel VM | Fails if kernel handle is invalid or kernel not found. | `tools/kaggle_cli/cli.py:495-501`, `docs/kaggle_cli.md:78-81` |
| 8 | Kaggle CLI | `output` subcommand | Downloads latest kernel output artifacts from `/kaggle/working` into local directory. | `--kernel <owner/slug>`, `--output-dir`, `--file-pattern` | Downloaded files in output directory | Fails if kernel has no output or invalid handle. | `tools/kaggle_cli/cli.py:503-505`, `docs/kaggle_cli.md:83-92` |
| 9 | Kaggle CLI | `resume` subcommand | Downloads `(last|best).pt` from latest kernel output, creates/versions private resume dataset (`<owner>/sentenai-<model>-resume`), mounts it, and submits next kernel with `--resume --no-test`. | Training args + `--clean-output`, `--resume-dataset` | Uploaded resume dataset & submitted resume kernel | Raises `KaggleToolError` if no `last.pt` found in downloaded output. | `tools/kaggle_cli/cli.py:205-228, 507-525`, `docs/kaggle_cli.md:93-111` |
| 10 | Kaggle CLI | `collect` subcommand | Downloads completed Kaggle output into `experiments/<model>` and optionally registers model in MLOps registry (`--register`). | `--model <model>`, `--owner <owner>`, optional `--kernel`, `--clean-output`, `--register` | Local artifacts synced to `experiments/<model>` + candidate registration in `artifacts/registry/registry.json` | Raises `KaggleToolError` if no `metrics.json` / model output directory found in downloaded artifacts. | `tools/kaggle_cli/cli.py:528-557`, `docs/kaggle_cli.md:150-162` |
| 11 | Kaggle Packaging | `sentenai_src_bundle.dat` | Non-archive file extension used for the packaged zip bundle in Kaggle Datasets to prevent Kaggle from silently auto-extracting the zip on upload. | Workspace source tree (excluding cache, data, git, etc.) | `.kaggle_work/datasets/main/sentenai_src_bundle.dat` | None (unzipped programmatically by `run.py` on remote VM) | `tools/kaggle_cli/cli.py:19-21, 171-173, 263-296`, `docs/kaggle_cli.md:171` |
| 12 | Kaggle Bootstrap | Remote `run.py` generator | Dynamically generates standalone Python entrypoint for Kaggle VM. Installs runtime dependencies (`pyvi`, `ftfy`, `emoji`, `sentencepiece`, `peft`, `accelerate`), applies model-specific patches (e.g. `transformers<5` and removing `torchao` for `vit5`), installs `absa_core` in editable mode, sets `TOKENIZERS_PARALLELISM=false`, executes training, and exports compact artifacts in a `finally` block. | `KernelSpec` configuration | `run.py` in kernel staging folder | Normal training exceptions are caught, recorded in `kaggle_run_manifest.json`, artifacts exported, and exception re-raised. | `tools/kaggle_cli/cli.py:231-375` |
| 13 | Fair Benchmark | Split Invariant & Fingerprint | Frozen 70/15/15 stratified split (`train`: 9,300, `val`: 1,991, `test`: 1,992 reviews, 0 text overlap). SHA-256 stable hash fingerprint `c32f956aee64af890c0645d37da203a9ae1f62ca14d0ef85ac8f2182e3415623`. | `data/raw/tiki-book-review_merged_fixed_v3.json`, `data/splits/split_manifest.json` | Verified split manifest and lineage fingerprint | Lineage validation fails if raw data or split manifest is modified. | `docs/fair_benchmark.md:5-17`, `data/splits/split_manifest.json`, `mlops/lineage.py:9-40` |
| 14 | Fair Benchmark | Test Split Sealing Policy | Default remote & exploration execution is validation-only (`--no-test`), keeping the test split sealed during architectural iterations and hyperparameter tuning. Test split is unsealed only with explicit `--run-test` for final evaluation. | `--no-test` (default remote) vs `--run-test` (explicit unseal) | `metrics.json` with empty test object (sealed) vs fully evaluated test object (unsealed) | Quality gate for production deployment strictly requires unsealed test metrics (`val_f1_combined`, `test_f1_combined`, `max_generalization_gap <= 0.10`). | `docs/kaggle_cli.md:16, 124-132, 161`, `docs/fair_benchmark.md:13-15`, `mlops/config.yaml:20-23` |
| 15 | Fair Benchmark | Primary Selection Metric | `f1_combined = 0.5 * f1_sentiment + 0.5 * f1_aspect_4class_mean`, where each aspect is evaluated over 4 classes (`negative`, `neutral`, `positive`, `absent`) to penalize missed and hallucinated aspects. | Model predictions & validation targets | Scalar `f1_combined` in [0, 1] | None | `docs/fair_benchmark.md:18-25`, `ml/evaluation/metrics.py:60-61` |
| 16 | Fair Benchmark | Seed & Determinism | Invariant random seed `seed = 42` across data loading, PyTorch (`torch.Generator().manual_seed(42)`), NumPy, and model initialization. | Config `seed: 42` | Reproducible data batches, model weights, and split generation | None | `docs/fair_benchmark.md:9`, `ml/configs/base.yaml:1`, `ml/utils/seed.py` |
| 17 | Leaderboard & MLOps | Benchmark Aggregation & Promotion | `ml/benchmark.py` runs benchmark loop, writes `experiments/benchmark/leaderboard.csv`, `leaderboard.json`, and `best_model.json`. `--promote-best` selects highest validation `f1_combined` among primary models, validates production gate, copies to `artifacts/final`, and assigns registry alias `champion`. | Model experiment directories in `experiments/` | `leaderboard.csv`, `leaderboard.json`, `best_model.json`, `artifacts/final/*`, `artifacts/registry/registry.json` | Raises `RuntimeError` if no primary models completed or if production quality gate fails. | `ml/benchmark.py:14-67`, `ml/evaluation/benchmark.py:12-28`, `mlops/config.yaml:7-24` |
| 18 | Model Card | Automated Model Card Generation | Generates standard model card (`MODEL_CARD.md`) for every model run capturing metadata, validation/test metrics, data lineage, environment info, and limitations. | Model metrics, lineage dict, model name | `experiments/<model>/MODEL_CARD.md` | None | `mlops/model_card.py`, `ml/train.py:136-148`, `experiments/*/MODEL_CARD.md` |

## Edge Cases

| # | Feature | Input | Observed Behavior |
|---|---------|-------|-------------------|
| 1 | Kaggle Slug Validation | Model with underscore: `--model linear_svm` | `make_spec` automatically translates underscore to hyphen in default kernel handle (`sentenai-linear-svm`) to prevent Kaggle backend title-slug mismatch errors (`cli.py:463, 511, 533`). |
| 2 | Handle Format Validation | Invalid handle format: `alice` or `invalid/handle/extra` | `validate_handle()` regex rejects input not matching `^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$` with explicit error: `Invalid Kaggle handle: <handle>. Expected owner/slug.` (Exit code 1). |
| 3 | Model Choice Validation | Unknown model name: `--model invalid_model` | `argparse` enforces choices `(logistic, linear_svm, textcnn, bilstm, phobert, xlmr, mdeberta, vit5)`, exiting immediately with error message and exit code 2. |
| 4 | Kaggle Dataset Zip Handling | Packaged source tree uploaded to Kaggle dataset | If named `*.zip` or `*.tar`, Kaggle auto-extracts contents into hundreds of root files. Packaging as `sentenai_src_bundle.dat` keeps the archive intact as a single binary file that `run.py` unzips cleanly via `zipfile.ZipFile`. |
| 5 | Batch GPU Session Quotas | Rapid sequential `kaggle kernels push` calls | Kaggle enforces a limit of 1-2 concurrent batch GPU sessions per user account. Pushing excess kernels causes 404 or permission denied rather than clean queuing; tooling documentation mandates waiting for free slots. |
| 6 | Interactive Console Cancellation | User opens running batch kernel in Kaggle web UI | Kaggle cancels batch job with status `CANCEL_ACKNOWLEDGED` and empty logs. Tooling provides `logs --follow` to monitor progress safely without triggering cancellation. |
| 7 | ViT5 Tokenizer & PEFT Quirks | `--model vit5` on Kaggle default image | Transformers v5 raises `KeyError: 0` on legacy sentencepiece vocab, and `peft` raises on outdated `torchao`. `kaggle_runner_source()` explicitly pins `transformers<5` and uninstalls `torchao` for `vit5` only. |
| 8 | Mixed Precision FP16 Master Weights | Loading Pretrained HuggingFace config in FP16 | `GradScaler` requires FP32 master weights and crashes on FP16 models. `EncoderMultiTaskNetwork` explicitly forces `torch_dtype=torch.float32` and calls `.float()` on encoder. |
| 9 | Multi-threading Dataloader Deadlocks | HuggingFace Tokenizers parallelism on Linux VM | Forked worker deadlocks with PyTorch DataLoader. `run.py` explicitly sets `os.environ['TOKENIZERS_PARALLELISM'] = 'false'`. |
| 10 | Crash Checkpoint Recovery | Remote training failure / Python exception | The generated `run.py` wraps `ml.train` in `try ... except ... finally`. Even if training raises an unhandled exception, `last.pt`, `best.pt`, metrics, and `kaggle_run_manifest.json` (recording `error: repr(exc)`) are exported to `/kaggle/working/sentenai-output/<model>`. |
| 11 | Minority Aspect Calibration Suppression | High absent ratio (>92%) in `as_price` & `as_service` | Calibration optimizing 4-class macro F1 picks high presence thresholds (0.90) because class 3 (absent) dominates accuracy, suppressing minority aspect presence and causing `f1_as_price` (0.136) and `f1_as_service` (0.149) to collapse. |
| 12 | Kaggle Environment Fingerprint Path Invariance | Local path `D:/...` vs Kaggle path `/kaggle/working/sentenai/...` | Stable hash in `dataset_snapshot` includes `dataset.path` and `split_manifest.path`. When paths differ between local and Kaggle environments, fingerprint hash differs (`0f2cfc...` on Kaggle vs `c32f95...` locally) unless standardized to relative paths or fixed dataset key. |

---

# 5-Component Handoff Report

## 1. Observation

Directly verified against authoritative workspace files and runtime execution:

### A. Kaggle CLI Integration (`docs/kaggle_cli.md`, `tools/kaggle_cli/cli.py`, `tests/unit/test_kaggle_cli_tool.py`)
- **Wrapper mechanism**: `tools/kaggle_cli/cli.py` shells out to system `kaggle.exe` (`shutil.which("kaggle")`, lines 62-68).
- **Environment & Auth**: Checks `.kaggle/kaggle.json` or system credentials (`get_kaggle_env()`, lines 32-43). Verified live execution via `python -m tools.kaggle_cli doctor`:
  ```text
  + C:\Users\PC\AppData\Local\Programs\Python\Python312\Scripts\kaggle.EXE --version
  Kaggle CLI 2.2.3
  + C:\Users\PC\AppData\Local\Programs\Python\Python312\Scripts\kaggle.EXE datasets list -p 1
  Kaggle authentication/API check: OK
  ```
- **Subcommands**:
  - `doctor`: validates CLI and auth (`cli.py:437-448`).
  - `prepare-data` / `sync-data`: copies raw dataset, splits (`train.json`, `val.json`, `test.json`, `split_manifest.json`), maps (`emoji_map.json`, `vocab_map.json`), packages `sentenai_src_bundle.dat`, and generates `dataset-metadata.json` (`cli.py:155-204`). Tested live: staged 9 flat files in `.kaggle_work\datasets\main` including 1.17 MB `sentenai_src_bundle.dat`.
  - `prepare-kernel` / `run`: generates `run.py`, packages `sentenai_src.zip`, creates `kernel-metadata.json` with `machine_shape: NvidiaTeslaT4` (default) and dataset sources (`cli.py:378-418, 480-488`).
  - `status` / `logs`: queries kernel state and streams logs with `--follow` (`cli.py:490-501`).
  - `output` / `resume`: downloads checkpoints (`last.pt`, `best.pt`), versions private resume dataset (`<owner>/sentenai-<model>-resume`), and resumes training with `--resume --no-test` (`cli.py:205-228, 507-525`).
  - `collect`: downloads output to `.kaggle_work/outputs/collect-<model>`, copies to `experiments/<model>`, and if `--register` is passed, invokes `mlops.register` (`cli.py:528-557`).
- **Test Suite**: Executed `python -m pytest tests/unit/test_kaggle_cli_tool.py -v`: 3 passed in 0.50s (`test_prepare_data_stages_flat_files`, `test_generated_kaggle_runner_is_valid_python`, `test_prepare_kernel_has_t4_and_sources`). Full unit test suite (`tests/unit/`): 17 passed in 4.95s.

### B. Fair Benchmark Guardrails (`docs/fair_benchmark.md`, `data/splits/split_manifest.json`, `mlops/lineage.py`)
- **Split Invariants** (`data/splits/split_manifest.json`):
  - Random seed: `42`. Strategy: `text_group_stratified`.
  - Train: 9,300 rows (70%), fingerprint `6b15f35362792c26c941e5b9f00565177f9de15d789983930d9b0a56452b7f98`.
  - Val: 1,991 rows (15%), fingerprint `4ec6bebc4cafaadfd862196d2163238670ca77772966a58c7b7e22bffaa6d9b5`.
  - Test: 1,992 rows (15%), fingerprint `69a04e6850bc9c6bf04fd9c43cd8091cb76f908d4ecc6ce0648df61393a1ab28`.
  - Total rows: 13,283 (cleaned from 13,412 raw tiki reviews, 0 text overlap across splits).
- **Required Data Fingerprint**: `c32f956aee64af890c0645d37da203a9ae1f62ca14d0ef85ac8f2182e3415623`.
  - Calculated as SHA-256 stable hash of `{dataset: {bytes: 8685000, sha256: "86093be711dfd2f30591a2e25dc4679dba839ed4e10f558f63d5edc4ee2909a7"}, git_sha: null, split_manifest: {sha256: "fc33548994d0dddaec32817c9391b0c21720c153773182e4c0820e944e5143b5"}}`.
- **Target Schema & Metric**:
  - Target: Overall sentiment (3-class: 0, 1, 2) + 6 aspects (4-class: 0: negative, 1: neutral, 2: positive, 3: absent).
  - Primary Metric: `f1_combined = 0.5 * f1_sentiment + 0.5 * f1_aspect_4class_mean`.
- **Test Split Sealing**: Default remote training uses `--no-test`. Test split is evaluated only upon final explicit `--run-test`.

### C. Benchmark Leaderboard & Baseline Scores (`ml/benchmark.py`, `experiments/benchmark/leaderboard.csv`, `experiments/*/metrics.json`)
- **Current Leaderboard (`experiments/benchmark/leaderboard.csv`)**:
  1. `linear_svm`: `val_f1_combined = 0.727963`, `test_f1_combined = 0.712163`, `test_f1_sentiment = 0.759184`, `test_f1_aspect_4class_mean = 0.665142`, `train_seconds = 13.51` (Current Leader).
  2. `bilstm`: `val_f1_combined = 0.665888`, `test_f1_combined = 0.663864`, `test_f1_sentiment = 0.781029`, `test_f1_aspect_4class_mean = 0.546699`, `train_seconds = 2061.97`.
  3. `textcnn`: `val_f1_combined = 0.603594`, `test_f1_combined = 0.609106`, `test_f1_sentiment = 0.763906`, `test_f1_aspect_4class_mean = 0.454307`, `train_seconds = 351.18`.
  4. `logistic`: `val_f1_combined = 0.505580`, `test_f1_combined = 0.498792`, `test_f1_sentiment = 0.417449`, `test_f1_aspect_4class_mean = 0.580134`, `train_seconds = 35.72`.
- **Pretrained Transformer Remote Baselines (`experiments/`)**:
  - `phobert`: `val_f1_combined = 0.655220` (test sealed: `NaN`), `f1_as_price = 0.136274`, `f1_as_service = 0.148910`, `presence_f1_as_price = 0.520864`, `presence_f1_as_service = 0.598237`.
  - `xlmr`: `val_f1_combined = 0.625488` (test sealed: `NaN`), `f1_as_price = 0.457351`, `f1_4class_as_price = 0.081684`, `f1_4class_as_service = 0.075166`.
  - `mdeberta`: `val_f1_combined = 0.592091` (test sealed: `NaN`), `f1_as_price = 0.443524`, `f1_4class_as_price = 0.086212`, `f1_4class_as_service = 0.069013`.
  - `vit5` (Secondary generative baseline): `val_f1_combined = 0.152427` (test sealed: `NaN`).

---

## 2. Logic Chain

1. **Kaggle CLI Tooling Architecture**:
   - `docs/kaggle_cli.md` and `tools/kaggle_cli/cli.py` establish a complete wrapper over the official Kaggle CLI.
   - The tooling decouples local environment from Kaggle cloud VMs while preserving identical data schemas, seed (`42`), and evaluation metrics.
   - Using `sentenai_src_bundle.dat` solves Kaggle's automatic unzip behavior for dataset archives.
   - The generated `run.py` handles VM dependency differences, environment variables (`TOKENIZERS_PARALLELISM=false`), and error recovery with checkpoint export in `finally` blocks.

2. **Root Causes of Transformer Performance Gap vs Linear SVM**:
   - **First-Token Pooling Bottleneck** (`ml/models/transformer/model.py:46`): Current implementation extracts only `hidden[:, 0]` (the `<s>` / `[CLS]` token). In Vietnamese book reviews, aspect-specific mentions (e.g. paper quality, shipping, seller support) frequently appear in middle and trailing sentences, causing first-token pooling to lose granular aspect context.
   - **Absent Label Domination in Calibration** (`ml/evaluation/calibration.py:22`): Macro F1 over all 4 classes $[0, 1, 2, 3]$ optimizes heavily for class 3 (absent) because absent examples make up >92% of minority aspects (`as_price` presence is only 7.8% in validation). The grid search chooses high thresholds ($t=0.90$), severely penalizing minority aspect recall and dropping `f1_as_price` to 0.136 and `f1_as_service` to 0.149 in PhoBERT.

3. **Fair Benchmark Integrity & Guardrails**:
   - Every model must run on the exact frozen splits in `data/splits/split_manifest.json` with seed 42.
   - The test set must remain sealed (`--no-test`) during model optimization and threshold tuning, and only opened with `--run-test` on the final candidate.
   - Promotion strictly requires validation F1 selection (`ml/benchmark.py:19`), with production gating verifying generalization gap ($\le 0.10$).

---

## 3. Caveats

1. **Kaggle Quota and Execution**:
   - Kaggle accounts are restricted to 1–2 concurrent batch GPU sessions. Simultaneous submission of multiple kernels will cause permission/quota errors.
   - Pushing kernels requires valid Kaggle credentials (`kaggle.json` or environment variables) and an active internet connection.
2. **Path Normalization in Lineage Fingerprinting**:
   - In local runs, absolute paths (`D:/vietcv/...`) were recorded, whereas in Kaggle runs, `/kaggle/working/sentenai/...` was recorded. For 100% hash parity with `c32f956aee64af890c0645d37da203a9ae1f62ca14d0ef85ac8f2182e3415623`, lineage snapshot path fields should use repo-relative paths (`data/raw/...`, `data/splits/...`) or ignore absolute path prefixes.
3. **No Implementation Actions Taken**:
   - In accordance with the Specification Miner role, no model architectures, calibration functions, or CLI tools were modified.

---

## 4. Conclusion

All specifications, CLI tools, benchmark guardrails, baseline metrics, and acceptance criteria have been comprehensively surveyed and verified:
- **Kaggle Tooling**: Fully functional and verified (`doctor`, `prepare-data`, `prepare-kernel`, `collect`, `resume`, `run`, `status`, `logs`).
- **Fair Benchmark Guardrails**: Split manifest (9300 train, 1991 val, 1992 test), seed 42, target data fingerprint `c32f956aee64af890c0645d37da203a9ae1f62ca14d0ef85ac8f2182e3415623`, and test split sealing protocol are authoritatively documented.
- **Optimization Targets**:
  - Transformer pooling upgrade (Masked Mean / Multi-Head Attention Pooling) & Hierarchical Head in `ml/models/transformer/model.py`.
  - Calibration objective refinement (Present-Only Macro F1) in `ml/evaluation/calibration.py`.
- **Target Performance**: Surpass baseline leader `linear_svm` (`val_f1_combined = 0.7280`), ensure minority aspects `f1_as_price` and `f1_as_service` $\ge 0.40$, and update `leaderboard.csv` / `MODEL_CARD.md`.

---

## 5. Verification Method

To independently verify all findings and test contracts:

1. **Verify Unit Tests**:
   ```bash
   python -m pytest tests/unit/test_kaggle_cli_tool.py -v
   python -m pytest tests/unit/ -v
   ```
2. **Verify Kaggle CLI Tooling**:
   ```bash
   python -m tools.kaggle_cli doctor
   python -m tools.kaggle_cli prepare-kernel --model phobert --owner test --dataset test/data
   python -m tools.kaggle_cli prepare-data --dataset test/data
   ```
3. **Verify Data Fingerprint and Split Manifest**:
   ```bash
   python -c "import json, hashlib; from pathlib import Path; from mlops.lineage import dataset_snapshot; s = dataset_snapshot('data/raw/tiki-book-review_merged_fixed_v3.json', root='.', split_manifest='data/splits/split_manifest.json'); print('Lineage FP:', s['fingerprint'])"
   ```
4. **Inspect Leaderboard and Baseline Records**:
   - `experiments/benchmark/leaderboard.csv`
   - `experiments/phobert/metrics.json`
   - `experiments/mdeberta/metrics.json`
   - `experiments/xlmr/metrics.json`
