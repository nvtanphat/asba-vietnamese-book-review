# Evaluation

All maintained models return seven probability matrices in a common contract: `(N,3)` overall sentiment and six `(N,4)` aspect distributions.

For each aspect, validation calibration searches a threshold on `P(present) = 1 - P(absent)`. If present, sentiment is `argmax` over classes 0–2; otherwise the prediction is class 3 (`absent`). The chosen threshold maximizes that aspect’s 4-class macro-F1 on validation, so false-positive aspects are penalized. Test thresholds are never fitted on test.

Reported metrics include:

- overall sentiment macro F1;
- aspect 4-class macro F1;
- aspect presence macro F1;
- aspect sentiment macro F1 on true-present labels;
- neutral F1 on true-present labels;
- six per-aspect present-only macro F1 scores;
- exact row match and flattened label accuracy;
- mean per-aspect 4-class macro-F1;
- `f1_combined = 0.5 * sentiment_macro_f1 + 0.5 * mean_per_aspect_4class_macro_f1`, the model-selection score;
- a legacy present-only combined score for historical comparison only.

## Paired significance check

Because every model predicts the exact same frozen test rows, `scripts/evaluation/significance.py MODEL_A MODEL_B` performs a paired bootstrap on the maintained `f1_combined` difference and reports a 95% interval. This is preferable to declaring a model superior from a tiny raw F1 difference alone.
