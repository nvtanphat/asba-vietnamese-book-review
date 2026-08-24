# Milestone M2 Review & Adversarial Audit Report: Calibration Algorithm Optimization

## 1. Observation

### 1.1 Codebase & Mathematical Implementation (`ml/evaluation/calibration.py`)
- **Mathematical Formulation**:
  `calibrate_absent_thresholds` optimizes per-aspect presence thresholds $t \in \text{grid}$ using full-dataset 3-class Present-Only Macro F1 (`labels=[0, 1, 2]`) combined with neutral sentiment protection weight ($w_{\text{neu}} = 0.15$):
  $$\text{Score}(t) = (1 - w_{\text{neu}}) \cdot F_{1,\text{pres}}(t) + w_{\text{neu}} \cdot F_{1,\text{neu}}(t)$$
  where:
  - $F_{1,\text{pres}}(t) = \text{precision\_recall\_fscore\_support}(y_{\text{true}}, \hat{y}_t, \text{labels}=[0, 1, 2], \text{average="macro"}, \text{zero\_division}=0)[2]$
  - $F_{1,\text{neu}}(t) = \text{precision\_recall\_fscore\_support}(y_{\text{true}}, \hat{y}_t, \text{labels}=[1], \text{average="macro"}, \text{zero\_division}=0)[2]$
- **Grid Range**: Default search grid is explicitly set to `np.linspace(0.10, 0.85, 31)` with step size $0.025$.
- **Fallback Mechanism**: Explicit fallback `best_t = 0.5` when `best_score <= 0.0` (zero-signal / all-absent ground truth).
- **Probability Decoding (`decode_probabilities`)**: Standardized to use `1.0 - probs[:, ABSENT_CLASS] >= threshold` for aspect presence detection and `probs[:, :3].argmax(axis=1)` for 3-class sentiment assignment, with strict shape validation (`p.shape[1] == 4`).

### 1.2 Test Execution Results
- **Unit Tests (`tests/unit/test_calibration.py`)**:
  - `test_calibration_contract`: PASSED
  - `test_minority_aspect_threshold_avoids_ceiling`: PASSED
  - `test_neutral_protection_influence`: PASSED
  - `test_decode_probabilities_defaults_and_validation`: PASSED
  - `test_zero_division_and_all_absent_edge_cases`: PASSED
  - `test_custom_grid_parameter`: PASSED
  - Total: 6/6 passed in 4.04s.
- **Full Repository Test Suite (`pytest tests/ -v`)**:
  - 140 passed in 20.14s (100% pass rate).
- **Code Linter (`ruff check`)**:
  - All checks passed clean.

### 1.3 End-to-End Validation on Real Dataset (`data/splits/`)
- Fitted `linear_svm` on `train.json` ($N=9300$) and calibrated on `val.json` ($N=1991$):
  - Calibrated thresholds: `{'as_content': 0.675, 'as_physical': 0.625, 'as_price': 0.675, 'as_packaging': 0.725, 'as_delivery': 0.650, 'as_service': 0.650}`
  - `f1_combined`: `0.728318`
  - `f1_aspect_presence`: `0.911162`
  - `f1_as_price`: `0.538850` ($> 0.40$)
  - `f1_as_service`: `0.660425` ($> 0.40$)
  - All thresholds safely remain in the interior range $[0.625, 0.725]$, avoiding the destructive $0.90$ ceiling collapse.

---

## 2. Logic Chain

1. **Elimination of the Absent Class Distorting Incentive**:
   - In 4-class macro F1, the absent class (`class 3`) constitutes >92% of minority aspect samples (`as_price`).
   - In the legacy formulation, predicting absent almost everywhere yielded $F_1^{(3)} \approx 0.95$, contributing $+0.237$ to the 4-class average regardless of present sentiment classification quality. This systematically drove thresholds to the search ceiling ($0.90$).
   - In the new 3-class present-only macro F1 formulation (`labels=[0, 1, 2]`), true absent samples predicted present are penalized as False Positives, while true present samples predicted absent are penalized as False Negatives. Class 3 does not contribute positive reward for absence. Pushing thresholds towards $0.85/0.90$ collapses recall and drives $F_{1,\text{pres}} \to 0$, forcing the optimizer to choose balanced interior thresholds.

2. **Preservation of Neutral Sentiment Recall**:
   - Neutral sentiment instances in minority aspects are scarce ($< 10$ examples in validation split).
   - Adding $w_{\text{neu}} = 0.15$ ensures that optimization does not discard neutral sentiment recall in favor of dominant positive/negative sentiment modes.

3. **Robustness & Edge-Case Safety**:
   - Zero division is safely handled by `zero_division=0` in sklearn calls.
   - All-absent and zero-signal inputs are safely handled by the `best_score <= 0.0` check, defaulting to $0.5$.

4. **Integrity Audit**:
   - No hardcoded constants, mock values, or dummy bypass logic exist in `ml/evaluation/calibration.py`.
   - The implementation is fully vectorized, modular, and compatible across all training loops (`ml/training/torch_text_trainer.py`, `ml/models/transformer/model.py`, `ml/train.py`).

---

## 3. Review & Adversarial Challenge Report

### Quality Review Summary
**Verdict**: **APPROVE**

### Verified Claims
- Mathematical formulation matches specification $\text{Score}(t) = (1 - w_{\text{neu}}) \cdot F_{1,\text{pres}}(t) + w_{\text{neu}} \cdot F_{1,\text{neu}}(t)$ $\rightarrow$ Verified via code inspection and mathematical analysis $\rightarrow$ **PASS**
- Default grid is `np.linspace(0.10, 0.85, 31)` with neutral weight $0.15$ $\rightarrow$ Verified in `ml/evaluation/calibration.py:20-22` $\rightarrow$ **PASS**
- Class 3 excluded from macro average, preventing ceiling collapse on minority aspects $\rightarrow$ Verified via simulated noisy distribution and real dataset validation $\rightarrow$ **PASS**
- All unit and regression tests pass $\rightarrow$ Verified independently via `pytest` (140/140 passed) $\rightarrow$ **PASS**
- Linter checks pass $\rightarrow$ Verified via `ruff check` $\rightarrow$ **PASS**

### Adversarial Stress-Test Results
- **Scenario 1: Extreme class imbalance (99% absent)**: Calibrated threshold remained interior ($t=0.50$ for zero-signal aspects, $t=0.10$ for clear positive signals) without numerical instability $\rightarrow$ **PASS**
- **Scenario 2: Perfect predictor (100% accuracy)**: Calibrated thresholds correctly set to minimal valid boundary ($t=0.10$), achieving $F_1 = 1.0$ $\rightarrow$ **PASS**
- **Scenario 3: Zero signal / Uniform random noise**: Zero division avoided, fallback triggered correctly ($t=0.50$) $\rightarrow$ **PASS**
- **Scenario 4: Real dataset split evaluation**: Minority aspects `as_price` ($0.5389$) and `as_service` ($0.6604$) both exceed the $\ge 0.40$ requirement $\rightarrow$ **PASS**

---

## 4. Caveats

- **Upstream Probability Calibration**: The threshold search optimizes based on model predicted probabilities. High-capacity neural models with overconfident or uncalibrated logits still depend on appropriate training loss functions (e.g. focal loss, label smoothing) to generate discriminative probabilities.

---

## 5. Conclusion

Milestone M2 (Calibration Algorithm Optimization) satisfies all mathematical, functional, architectural, and fair benchmark requirements with zero integrity violations.

**Explicit Verdict**: **APPROVE**

---

## 6. Verification Method

To independently verify this review:
1. Run calibration unit tests:
   ```powershell
   python -m pytest tests/unit/test_calibration.py -v
   ```
2. Run full test suite:
   ```powershell
   python -m pytest tests/ -v
   ```
3. Run linter:
   ```powershell
   python -m ruff check ml/evaluation/calibration.py tests/unit/test_calibration.py
   ```
