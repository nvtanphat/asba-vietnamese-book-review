# absa-core

Shared Vietnamese **Aspect-Based Sentiment Analysis (ABSA)** runtime for SentenAI Unified.

This package is the single source of truth for preprocessing and production inference. It supports:

- the promoted artifact produced by the unified fair benchmark (`UnifiedArtifactPredictor`),
- classical TF-IDF models, TextCNN, BiLSTM, and pretrained encoder artifacts,
- the legacy PhoBERT predictor/ONNX path as a compatibility fallback,
- the same packaged emoji/vocabulary normalization maps used by training and serving.

Target schema:

- overall sentiment: `0=negative`, `1=neutral`, `2=positive`;
- six aspects: `as_content`, `as_physical`, `as_price`, `as_packaging`, `as_delivery`, `as_service`;
- benchmark aspect targets use four classes: `0=negative`, `1=neutral`, `2=positive`, `3=absent`.

Production promotion writes the winner to `artifacts/final/`. FastAPI prefers that artifact and falls back to the legacy predictor when no promoted artifact exists.
