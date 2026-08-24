## 2026-08-24T00:47:33Z
You are Worker for Milestone M2 (Minority Aspect Calibration Algorithm Optimization).
Working directory: D:\vietcv\SentenAI-Unified\.agents\worker_m2
Authoritative User Request: D:\vietcv\SentenAI-Unified\.agents\ORIGINAL_REQUEST.md
PROJECT.md: D:\vietcv\SentenAI-Unified\PROJECT.md
Calibration Blueprint: D:\vietcv\SentenAI-Unified\.agents\explorer_survey_calib\handoff.md
Workspace root: D:\vietcv\SentenAI-Unified

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your write ownership:
- `ml/evaluation/calibration.py`
- `tests/unit/test_calibration.py`

Tasks:
1. Review `D:\vietcv\SentenAI-Unified\.agents\explorer_survey_calib\handoff.md`.
2. Update `ml/evaluation/calibration.py` (`calibrate_absent_thresholds`) to implement the full-dataset 3-class Present-Only Macro F1 objective (`labels=[0, 1, 2]`) with neutral protection ($w_{\text{neu}} = 0.15$) and grid `np.linspace(0.10, 0.85, 31)`.
3. Update and enhance `tests/unit/test_calibration.py` to verify:
   - Present-Only Macro F1 calibration behavior
   - Minority aspect threshold optimization avoiding the 0.90 ceiling
   - Decode probabilities integration
   - Zero division and edge cases
4. Run all unit and smoke tests: `python -m pytest tests/unit tests/smoke -v` (and `pytest tests/ -v`).
5. Write your detailed completion and verification report to `D:\vietcv\SentenAI-Unified\.agents\worker_m2\handoff.md`. Send a brief notification message.
