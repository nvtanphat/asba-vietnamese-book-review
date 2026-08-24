# Model Card — vit5

Generated: 2026-08-24T15:18:47.670769+00:00

## Intended use
Vietnamese Tiki book-review aspect-based sentiment analysis (ABSA). The production task predicts overall sentiment and six aspect sentiments.

## Evaluation

```json
{
  "val_f1_combined": 0.737535,
  "test_f1_combined": 0.729385,
  "test_f1_sentiment": 0.756657,
  "test_f1_aspect_4class_mean": 0.702113
}
```

## Data lineage

```json
{
  "created_at": "2026-08-24T13:17:45.497926+00:00",
  "dataset": {
    "path": "/kaggle/working/sentenai/data/raw/tiki-book-review_merged_fixed_v3.json",
    "bytes": 8685000,
    "sha256": "86093be711dfd2f30591a2e25dc4679dba839ed4e10f558f63d5edc4ee2909a7"
  },
  "git_sha": null,
  "split_manifest": {
    "path": "/kaggle/working/sentenai/data/splits/split_manifest.json",
    "sha256": "fc33548994d0dddaec32817c9391b0c21720c153773182e4c0820e944e5143b5"
  },
  "environment": {
    "python": "3.12.13",
    "platform": "linux",
    "executable": "/usr/bin/python3"
  },
  "fingerprint": "0f2cfce98f90e746d5a8a20f0a50fbdb04e56cc53271c8cd4ed056810ac6f394"
}
```

## Limitations
The dataset is domain-specific, label/aspect distributions are imbalanced, and production drift must be monitored before reusing the model for other commerce domains.

## Notes
Generated automatically by the unified training pipeline.
