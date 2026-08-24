# Dataset

The unified repository ships the user-provided `data/raw/tiki-book-review_merged_fixed_v3.json` as the single raw source for reproducibility.

## Raw schema and observed counts

| Field | Meaning |
|---|---|
| `review_id` | Tiki review identifier |
| `content` | Review text |
| `sentiment` | Overall sentiment: 0 negative, 1 neutral, 2 positive |
| `as_content` | Content sentiment or null |
| `as_physical` | Physical book quality sentiment or null |
| `as_price` | Price sentiment or null |
| `as_packaging` | Packaging sentiment or null |
| `as_delivery` | Delivery sentiment or null |
| `as_service` | Service sentiment or null |

Raw file inspection in this merge found **13,412 rows**, **2,009 products**, **13,308 rows with an overall sentiment label**, and six rows with missing review text. Aspect presence is strongly imbalanced: content 5,276; physical 7,170; price 970; packaging 3,206; delivery 3,517; service 2,373.

Aspect null is converted to the explicit class `3 = absent` only inside the modeling layer. The original raw file is never overwritten.

## Frozen split

`python -m ml.data.split` creates a single 70/15/15 train/validation/test split. The default `text_group_stratified` strategy groups normalized duplicate review texts before splitting so the same normalized review cannot cross split boundaries. A `split_manifest.json` records the seed, row counts, label distributions and SHA-256 fingerprints.

An optional `product_group` strategy exists for a stricter unseen-product generalization experiment; it must be treated as a different benchmark and must not be mixed with the default leaderboard.
