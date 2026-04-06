import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictor import ABSAPredictor, ASPECT_COLS


ABSENT_LABEL = 3
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ABSA model with notebook-style metrics (05_phobert_balance_experiment)."
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/test_clean.json",
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument(
        "--model-variant",
        default="phobert",
        choices=["phobert", "bilstm-phobert"],
        help="Model variant to evaluate.",
    )
    parser.add_argument(
        "--model-id",
        default="hoangloc112/ABSA-TIKI-BOOK",
        help="HF repo id or local model directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save JSON summary.",
    )
    return parser.parse_args()


def _predict_in_batches(predictor: ABSAPredictor, texts: list[str], batch_size: int) -> list[dict]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    out: list[dict] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        out.extend(predictor.predict(texts[start:end]))
        print(f"Inference: {end}/{total}")
    return out


def _prepare_true_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if "content" not in df.columns or "sentiment_llm" not in df.columns:
        raise ValueError("Dataset must contain columns: 'content', 'sentiment_llm'.")

    sent_numeric = pd.to_numeric(df["sentiment_llm"], errors="coerce")
    valid_mask = sent_numeric.notna()
    if not valid_mask.all():
        dropped = int((~valid_mask).sum())
        print(f"[WARN] Dropping {dropped} rows with invalid sentiment_llm.")
        df = df.loc[valid_mask].reset_index(drop=True)
        sent_numeric = sent_numeric.loc[valid_mask]

    true_sent = sent_numeric.astype(int).to_numpy()
    true_asps = (
        df[ASPECT_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(ABSENT_LABEL)
        .astype(int)
        .to_numpy()
    )
    return df, true_sent, true_asps


def _decode_predictions(predictions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = len(predictions)
    pred_sent = np.zeros(n, dtype=int)
    pred_asps = np.full((n, len(ASPECT_COLS)), ABSENT_LABEL, dtype=int)

    for i, pred in enumerate(predictions):
        pred_sent[i] = int(pred["overall"])
        for j, col in enumerate(ASPECT_COLS):
            v = int(pred["aspects"][col])
            pred_asps[i, j] = ABSENT_LABEL if v == -1 else v
    return pred_sent, pred_asps


def compute_notebook_style_metrics(
    true_sent: np.ndarray,
    pred_sent: np.ndarray,
    true_asps: np.ndarray,
    pred_asps: np.ndarray,
) -> dict:
    # Same metric recipe as notebooks/05_phobert_balance_experiment.ipynb
    f1_sent = precision_recall_fscore_support(
        true_sent,
        pred_sent,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )[2]

    true_flat = true_asps.flatten()
    pred_flat = pred_asps.flatten()

    f1_aspect_all = precision_recall_fscore_support(
        true_flat,
        pred_flat,
        labels=[0, 1, 2, ABSENT_LABEL],
        average="macro",
        zero_division=0,
    )[2]

    present_mask = true_flat != ABSENT_LABEL
    f1_aspect_present = 0.0
    f1_aspect_neutral_present = 0.0
    if present_mask.any():
        f1_aspect_present = precision_recall_fscore_support(
            true_flat[present_mask],
            pred_flat[present_mask],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )[2]
        f1_aspect_neutral_present = precision_recall_fscore_support(
            true_flat[present_mask],
            pred_flat[present_mask],
            labels=[1],
            average="macro",
            zero_division=0,
        )[2]

    asp_f1s = {}
    for i, col in enumerate(ASPECT_COLS):
        mask = true_asps[:, i] != ABSENT_LABEL
        if mask.any():
            f1_i = precision_recall_fscore_support(
                true_asps[:, i][mask],
                pred_asps[:, i][mask],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )[2]
        else:
            f1_i = 0.0
        asp_f1s[f"f1_{col}"] = round(float(f1_i), 4)

    acc = accuracy_score(
        np.concatenate([true_sent, true_flat]),
        np.concatenate([pred_sent, pred_flat]),
    )

    return {
        "f1_sentiment": round(float(f1_sent), 4),
        "f1_aspect_all": round(float(f1_aspect_all), 4),
        "f1_aspect_present": round(float(f1_aspect_present), 4),
        "f1_aspect_neutral_present": round(float(f1_aspect_neutral_present), 4),
        "f1_combined": round(float(0.5 * f1_sent + 0.5 * f1_aspect_present), 4),
        "accuracy": round(float(acc), 4),
        **asp_f1s,
    }


def _print_notebook_style_reports(
    true_sent: np.ndarray,
    pred_sent: np.ndarray,
    true_asps: np.ndarray,
    pred_asps: np.ndarray,
) -> dict:
    print("\n=== OVERALL SENTIMENT (TEST SET) ===")
    overall_text = classification_report(
        true_sent,
        pred_sent,
        labels=[0, 1, 2],
        target_names=SENTIMENT_LABELS,
        zero_division=0,
    )
    print(overall_text)

    print("=== 6 ASPECTS: present only (TEST SET) ===")
    present_mask = true_asps.flatten() != ABSENT_LABEL
    if present_mask.any():
        aspect_text = classification_report(
            true_asps.flatten()[present_mask],
            pred_asps.flatten()[present_mask],
            labels=[0, 1, 2],
            target_names=SENTIMENT_LABELS,
            zero_division=0,
        )
        print(aspect_text)
        aspect_json = classification_report(
            true_asps.flatten()[present_mask],
            pred_asps.flatten()[present_mask],
            labels=[0, 1, 2],
            target_names=SENTIMENT_LABELS,
            zero_division=0,
            output_dict=True,
        )
    else:
        print("No present aspects in ground truth.")
        aspect_json = {}

    overall_json = classification_report(
        true_sent,
        pred_sent,
        labels=[0, 1, 2],
        target_names=SENTIMENT_LABELS,
        zero_division=0,
        output_dict=True,
    )

    return {
        "overall_sentiment_report": overall_json,
        "aspects_present_only_report": aspect_json,
    }


def run_evaluation(
    data_path: str = "data/processed/test_clean.json",
    model_variant: str = "phobert",
    model_id: str = "hoangloc112/ABSA-TIKI-BOOK",
    batch_size: int = 64,
) -> dict:
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    df_raw = pd.read_json(data_file)
    df_eval, true_sent, true_asps = _prepare_true_labels(df_raw)
    texts = df_eval["content"].astype(str).tolist()

    print(f"Evaluating model_variant='{model_variant}'")
    print(f"Data: {data_file} | rows={len(df_eval)}")

    predictor = ABSAPredictor(model_id=model_id, model_variant=model_variant)
    predictions = _predict_in_batches(predictor, texts, batch_size=batch_size)
    pred_sent, pred_asps = _decode_predictions(predictions)

    metrics = compute_notebook_style_metrics(true_sent, pred_sent, true_asps, pred_asps)
    reports = _print_notebook_style_reports(true_sent, pred_sent, true_asps, pred_asps)

    print("=== NOTEBOOK-STYLE METRICS ===")
    print(metrics)

    return {
        "meta": {
            "data_path": str(data_file),
            "n_rows": len(df_eval),
            "model_variant": model_variant,
            "model_id": model_id,
            "batch_size": batch_size,
        },
        "metrics": metrics,
        "reports": reports,
    }


def main():
    args = parse_args()
    summary = run_evaluation(
        data_path=args.data_path,
        model_variant=args.model_variant,
        model_id=args.model_id,
        batch_size=args.batch_size,
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary: {out_path}")


if __name__ == "__main__":
    main()
