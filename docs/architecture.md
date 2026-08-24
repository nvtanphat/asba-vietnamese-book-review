# Unified Architecture

SentenAI now combines the research repository and dashboard repository into one lifecycle instead of two drifting codebases.

```text
Tiki / provided dataset
        ↓
 scraper + data/raw
        ↓
 absa_core.preprocessing  ← single source of truth
        ↓
 data/splits + manifest
        ↓
 ml/models registry
 ├── Logistic Regression
 ├── Linear SVM
 ├── TextCNN
 ├── BiLSTM
 ├── PhoBERT
 ├── XLM-R
 ├── mDeBERTa-v3
 └── ViT5 + LoRA (secondary generative)
        ↓
 shared calibration + evaluator
        ↓
 experiments/benchmark
        ↓ validation selection
 artifacts/final
        ↓
 absa_core UnifiedArtifactPredictor
        ↓
 FastAPI
        ↓
 Next.js dashboard
```

The older specialized PhoBERT trainer is retained under `ml/legacy_phobert/` for auditability, not for the fair leaderboard.
