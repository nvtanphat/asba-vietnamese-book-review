# Fair Benchmark Protocol

The original research notebooks used different code paths, feature engineering, data handling and evaluation details. Their scores are therefore retained only as historical experiments and are **not** used as direct evidence that one architecture is better than another.

The unified benchmark enforces these invariants:

1. One immutable raw dataset and one frozen split manifest.
2. One shared semantic cleaning pipeline from `absa_core.preprocessing`.
3. One seed (`42` by default) and deterministic dataloader setup where supported.
4. One target schema: overall sentiment is 3-class; each of six aspects is 4-class (`negative`, `neutral`, `positive`, `absent`).
5. One evaluator and one primary metric definition for every model.
6. Per-aspect presence thresholds are calibrated on validation only.
7. Hyperparameter tuning uses the same trial budget per model (default 20 Optuna trials) and never opens the test set.
8. The production winner is selected by **validation `f1_combined`**, never test score.
9. Test results are reported for comparison after model/config selection.
10. Model-specific tokenizer adapters are permitted because they are intrinsic to pretrained architectures; semantic text cleaning and labels remain identical. The word-level TextCNN/BiLSTM baselines and PhoBERT use Vietnamese word segmentation (`PyVi`); XLM-R/mDeBERTa keep their native subword tokenization. This adapter is treated as tokenizer-native processing, not semantic data cleaning.

## Primary score

`f1_combined = 0.5 * overall_sentiment_macro_f1 + 0.5 * mean(per_aspect_4class_macro_f1)`

Each aspect is evaluated over `negative / neutral / positive / absent`. This directly penalizes both missed aspects and false-positive (hallucinated) aspects. Present-only sentiment F1 is still reported as a diagnostic and as a legacy-comparison metric, but it is not the maintained selection objective.

The leaderboard also reports 4-class aspect F1, aspect-presence F1, per-aspect present-only F1, exact-match accuracy, parameter count and train wall-clock time.

## Primary vs secondary benchmark

Logistic Regression, Linear SVM, TextCNN, BiLSTM, PhoBERT, XLM-R and mDeBERTa share the same discriminative 7-head task formulation and are eligible for automatic production promotion. ViT5 is retained in the leaderboard as a **secondary generative comparison**. It uses the same split and evaluator, but its autoregressive objective and LoRA training are not formulation/compute-equivalent, so it is not automatically promoted as the primary winner.
