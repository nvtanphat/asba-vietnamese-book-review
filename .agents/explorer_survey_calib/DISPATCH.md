## 2026-08-24T00:22:16Z
You are Calibration Explorer investigating SentenAI-Unified.
Working directory: D:\vietcv\SentenAI-Unified\.agents\explorer_survey_calib
Authoritative User Request: D:\vietcv\SentenAI-Unified\.agents\ORIGINAL_REQUEST.md
Workspace root: D:\vietcv\SentenAI-Unified

Your mission is to explore and survey:
1. Current threshold calibration algorithms in `ml/evaluation/calibration.py` (specifically `calibrate_absent_thresholds`, `find_best_thresholds`, or related functions).
2. How the objective function is currently calculated and why it gets dominated by the accuracy of the absent label.
3. How to formulate and optimize the objective for **Present-Only Macro F1** across aspect sentiments, specifically protecting minority aspects (`as_price`, `as_service`).
4. How calibrated thresholds are stored, exported, loaded, and evaluated during validation and test runs.
5. Exact file paths, function signatures, data flow, metrics calculation in `ml/evaluation/`, and concrete recommendations.

Write your detailed findings report to `D:\vietcv\SentenAI-Unified\.agents\explorer_survey_calib\handoff.md`. When finished, send a brief message with the handoff path.
