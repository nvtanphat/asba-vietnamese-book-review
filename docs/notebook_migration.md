# Notebook Migration

No `.ipynb` file is part of the unified repository. Every original notebook was converted to a Python audit script under `scripts/migrated_notebooks/`, while its reusable logic was moved into maintained modules.

| Original notebook | Maintained destination |
|---|---|
| `00_preprocessing_debug.ipynb` | `scripts/data/debug_preprocessing.py` + `absa_core.preprocessing` |
| `01_before_after_preprocessing.ipynb` | `scripts/data/compare_preprocessing.py` |
| `02_eda_detailed_visualization.ipynb` | `scripts/analysis/run_eda.py` + `absa_core.analysis` |
| `03_0_baseline_logistic_regression.ipynb` | `ml/models/logistic/` + shared sklearn multi-task wrapper |
| `03_1_baseline_5fold_cv.ipynb` | `scripts/training/cross_validate.py` |
| `04_0_bilstm_multitask.ipynb` | `ml/models/bilstm/` |
| `04_1_bilstm_embedding.ipynb` | historical code preserved as migrated `.py`; embedding ablation is not part of the fair primary benchmark |
| `04_2_bilstm_word_segmenter.ipynb` | historical code preserved as migrated `.py`; segmentation is treated as a separate ablation |
| `05_0_phobert_data_balance.ipynb` | shared transformer model + common evaluator/calibration; old specialized recipe archived in `ml/legacy_phobert/` |
| `06_0_vit5_generative_absa.ipynb` | `ml/models/vit5/` |

The converted scripts intentionally preserve original cell code for traceability. They are excluded from maintained lint rules because notebook-era exploratory constructs are not the production source of truth.
