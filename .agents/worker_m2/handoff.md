# Milestone M2 Handoff Report: Minority Aspect Calibration Algorithm Optimization

## 1. Observation

### 1.1 Root Cause Verification
- In the baseline calibration implementation (`ml/evaluation/calibration.py`), threshold calibration maximized a 4-class macro F1 score across classes `[0, 1, 2, 3]`:
  $$\text{Macro-F1} = \frac{1}{4}\left(F_1^{(0)} + F_1^{(1)} + F_1^{(2)} + F_1^{(3)}\right)$$
- Because the dataset is heavily skewed towards the absent class (`class 3`, reaching 92.1% in `as_price` and 82.0% in `as_service`), neural models outputting slight probability noise on absent samples caused 4-class macro F1 to reward pushing the presence threshold $t$ up to the ceiling $0.90$ to maximize $F_1^{(3)}$, at the catastrophic expense of present sentiment classes ($F_1^{(0)}, F_1^{(1)}, F_1^{(2)}$).
- This collapsed `f1_as_price` to 0.136 and `f1_as_service` to 0.148 in PhoBERT.

### 1.2 Implemented Changes in `ml/evaluation/calibration.py`
- Refactored `calibrate_absent_thresholds` to compute full-dataset 3-class Present-Only Macro F1 (`labels=[0, 1, 2]`) with neutral protection weight ($w_{\text{neu}} = 0.15$):
  $$\text{Score}(t) = (1 - w_{\text{neu}}) \cdot F_{1,\text{pres}}(t) + w_{\text{neu}} \cdot F_{1,\text{neu}}(t)$$
  where:
  - $F_{1,\text{pres}}(t) = \text{precision\_recall\_fscore\_support}(y_{\text{true}}, \hat{y}_t, \text{labels}=[0, 1, 2], \text{average="macro"}, \text{zero\_division}=0)[2]$
  - $F_{1,\text{neu}}(t) = \text{precision\_recall\_fscore\_support}(y_{\text{true}}, \hat{y}_t, \text{labels}=[1], \text{average="macro"}, \text{zero\_division}=0)[2]$
- Default search grid updated to `np.linspace(0.10, 0.85, 31)`.
- Degenerate/zero-signal fallback explicitly set to `0.5` if `best_score <= 0.0`.
- Standardized `decode_probabilities` to use `ABSENT_CLASS` (3) and default fallback `0.5`.

### 1.3 Implemented Unit Tests in `tests/unit/test_calibration.py`
- `test_calibration_contract`: Verifies calibration return types, keys matching `ASPECT_COLS`, and decoded output array dimensions $(N, 7)$.
- `test_minority_aspect_threshold_avoids_ceiling`: Simulates 90% absent minority distribution with noisy logits; asserts calibrated threshold avoids ceiling/floor ($0.30 \le t \le 0.75$) and yields minority aspect F1 $> 0.40$.
- `test_neutral_protection_influence`: Verifies that `neutral_weight` parameter properly influences optimization.
- `test_decode_probabilities_defaults_and_validation`: Tests default threshold behavior (`thresholds=None`) and invalid shape exception handling (`ValueError` for aspect probabilities without 4 classes).
- `test_zero_division_and_all_absent_edge_cases`: Tests all-absent ground truth and uniform probabilities, ensuring zero-division safety and fallback to 0.5.
- `test_custom_grid_parameter`: Verifies custom threshold grids are strictly respected.

---

## 2. Logic Chain

1. **Step 1: Metric Alignment on Full Dataset**
   - By computing $F_1$ over `labels=[0, 1, 2]` on the full dataset, any sample where $y=3$ (absent) is predicted as present ($\hat{y} \in \{0, 1, 2\}$) is counted as a **false positive** for class $\hat{y}$.
   - Any sample where $y \in \{0, 1, 2\}$ (present) is predicted as absent ($\hat{y}=3$) is counted as a **false negative** for class $y$.
   - The absent class `3` is excluded from the macro average, removing the artificial $+0.25 \cdot F_1^{(3)}$ incentive that previously pushed thresholds to $0.90$.

2. **Step 2: Neutral Protection Factor**
   - Neutral sentiment instances in minority aspects are extremely rare (e.g., only 6 instances in validation set for `as_price` and `as_service`).
   - The $w_{\text{neu}} = 0.15$ weighting ensures that threshold selection maintains sensitivity towards neutral mentions rather than sacrificing them entirely.

3. **Step 3: Empirical Validation on Ground Truth Dataset**
   - Evaluated on validation split (`val.json`, $N=1991$) using Linear SVM probabilities:
     - Calibrated thresholds: `{'as_content': 0.675, 'as_physical': 0.625, 'as_price': 0.675, 'as_packaging': 0.725, 'as_delivery': 0.65, 'as_service': 0.65}`
     - Minority aspect F1 results: `f1_as_price = 0.5389`, `f1_as_service = 0.6604` (both $> 0.40$).
     - Overall metrics: `f1_combined = 0.7283`, `f1_aspect_presence = 0.9112`, `f1_sentiment = 0.7665`.

---

## 3. Caveats

1. **Model Probability Quality**: The calibration algorithm optimizes thresholds based on model output probabilities. Models with poor probability discrimination or uncalibrated logits will benefit from threshold tuning, but model training quality (loss weighting, architecture) remains crucial.
2. **Evaluation Policy**: Calibration thresholds are strictly fitted on validation data (`val_probs`, `val_y`) and applied to test sets without leaking test distribution information.

---

## 4. Conclusion

Milestone M2 requirements are fully completed:
1. `ml/evaluation/calibration.py` implements the 3-class Present-Only Macro F1 objective with neutral protection ($w_{\text{neu}} = 0.15$) and grid `np.linspace(0.10, 0.85, 31)`.
2. `tests/unit/test_calibration.py` provides complete, rigorous test coverage for contract, minority threshold ceiling avoidance, neutral weighting, error handling, and edge cases.
3. 100% of repository tests pass (140/140 unit and smoke tests).
4. Code passes all linting and syntax checks (`ruff check`).

---

## 5. Verification Method

### 5.1 Run Calibration Unit Tests
```powershell
python -m pytest tests/unit/test_calibration.py -v
```

### 5.2 Run Full Unit and Smoke Test Suite
```powershell
python -m pytest tests/unit tests/smoke -v
```

### 5.3 Run Linter
```powershell
python -m ruff check ml/evaluation/calibration.py tests/unit/test_calibration.py
```
