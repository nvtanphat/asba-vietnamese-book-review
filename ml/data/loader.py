from __future__ import annotations

from pathlib import Path
import pandas as pd

from .schema import ASPECT_COLS, TARGET_COLS, ABSENT_CLASS


def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def prepare_frame(frame: pd.DataFrame, text_col: str | None = None) -> pd.DataFrame:
    text_col = text_col or ("text" if "text" in frame.columns else "content")
    required = [text_col, "sentiment", *ASPECT_COLS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_meta = [c for c in ["review_id", "product_id", "product_name", "category", "rating", "created_at"] if c in frame.columns]
    out = frame[[*keep_meta, text_col, *TARGET_COLS]].copy()
    out = out.rename(columns={text_col: "text"})
    out["text"] = out["text"].astype("string").str.strip()
    out = out[out["text"].notna() & out["text"].ne("")]

    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce")
    out = out[out["sentiment"].isin([0, 1, 2])]
    out["sentiment"] = out["sentiment"].astype(int)

    for col in ASPECT_COLS:
        values = pd.to_numeric(out[col], errors="coerce").fillna(ABSENT_CLASS).astype(int)
        out[col] = values.where(values.isin([0, 1, 2, 3]), ABSENT_CLASS)
    return out.reset_index(drop=True)


def load_splits(data_root: str | Path = "data/splits") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_root)
    candidates = {
        "train": [root / "train.json", root / "train_clean.json"],
        "val": [root / "val.json", root / "val_clean.json"],
        "test": [root / "test.json", root / "test_clean.json"],
    }
    frames = []
    for split, paths in candidates.items():
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Cannot find {split} split under {root}")
        frames.append(prepare_frame(read_frame(path)))
    return tuple(frames)  # type: ignore[return-value]
