# Worker M2 Context
Milestone M2: Calibration Algorithm Optimization for Minority Aspects
Authoritative User Request: D:\vietcv\SentenAI-Unified\.agents\ORIGINAL_REQUEST.md
PROJECT.md: D:\vietcv\SentenAI-Unified\PROJECT.md
Calibration Survey Blueprint: D:\vietcv\SentenAI-Unified\.agents\explorer_survey_calib\handoff.md

Task:
1. Update `ml/evaluation/calibration.py`: Replace `calibrate_absent_thresholds` with the full-dataset 3-class Present-Only Macro F1 objective (`labels=[0, 1, 2]`) with neutral protection ($w_{\text{neu}} = 0.15$) and finer grid search (`np.linspace(0.10, 0.85, 31)`).
2. Update `tests/unit/test_calibration.py` to test the new Present-Only calibration function across regular and minority aspect scenarios, ensuring thresholds do not collapse to 0.90.
3. Run all tests: `python -m pytest tests/ -v`.
