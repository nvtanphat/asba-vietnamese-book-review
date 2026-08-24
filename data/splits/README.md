# Frozen split output

Run `make prepare` to generate `train.json`, `val.json`, `test.json` and `split_manifest.json` from the immutable raw v3 file. The row files are intentionally ignored by Git; the manifest can be committed to freeze a benchmark release.
