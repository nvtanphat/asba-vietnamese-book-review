# Model Architectures

## Classical baselines

**Logistic Regression** and **Linear SVM** share the exact same `FeatureUnion`: word TF-IDF (1–2 grams) plus character TF-IDF (3–5 grams). Seven independent target estimators are trained on the same sparse matrix. Character n-grams are intentionally included because Tiki reviews contain typos, teencode and code-switching.

## Neural baselines

**TextCNN** uses PyVi word segmentation, a learned embedding, parallel 1-D convolutions and global max pooling before seven classification heads. **BiLSTM** uses the same PyVi word-level input, a learned embedding, bidirectional LSTM, masked mean/max pooling and the same seven-head output schema. These are clean neural baselines; the more complex PhoBERT+BiLSTM and embedding/segmenter ablations from the original notebooks remain preserved in `scripts/migrated_notebooks/`.

## Pretrained encoders

**PhoBERT**, **XLM-RoBERTa-base** and **mDeBERTa-v3-base** use one common architecture wrapper: pretrained encoder → first-token pooled representation → dropout → seven classification heads. Only the pretrained backbone/tokenizer adapter changes. PhoBERT uses PyVi word segmentation to match its word-segmented pretraining regime; XLM-R and mDeBERTa use native subword tokenization. This removes the previous architecture-code confound when comparing encoders.

## ViT5

**ViT5** converts the seven targets into a deterministic text format such as `sentiment=positive; as_content=positive; ...`. It is fine-tuned with LoRA by default and parsed back into the same seven labels for evaluation.
