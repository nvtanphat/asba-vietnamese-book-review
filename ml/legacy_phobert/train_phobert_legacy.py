"""
Train PhoBERT ABSA — best experiment: clean_joint_balanced_focal
(from 05-phobert-balance-experiment-under1mb.ipynb).
Calibrated val F1_combined=0.7976, test F1_combined=0.7924.

Usage:
    python ml/train.py
    python ml/train.py --data-root /path/to/splits --output-dir ./runs

Evaluate errors / confusion matrices after training:
    python ml/scripts/eval_phobert.py --model-dir <saved_dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

from core import (
    ABSAModel, ABSAMultiTaskTrainer,
    BEST_EXPERIMENT, build_joint_label_sampler, build_training_args, build_weight_tensors,
    calibrate_presence_thresholds, compute_aspect_loss_multipliers, compute_metrics,
    decode_with_thresholds, load_splits, metrics_from_decoded, predict_review,
    seed_everything, tokenize_frame,
)
from core.config import (
    ASPECT_COLS, BASE_FOCAL_GAMMA, DEFAULT_DATA_ROOT, FOCAL_CONFIG_STANDARD, MODEL_NAME,
    SAMPLER_TEMPERATURE,
)


def run_experiment(cfg, train_df, sent_w, pres_w, s2_w, aspect_mult, tokenizer, val_encoded, device):
    """Train one experiment config, save checkpoint, return (result, model, tokenizer)."""
    train_encoded = tokenize_frame(train_df, tokenizer)
    model         = ABSAModel.from_pretrained(MODEL_NAME).to(device)
    output_dir    = f"./absa_results/{cfg['name']}"
    args          = build_training_args(output_dir)

    sent_weights  = None
    if cfg["use_class_weights"] and cfg.get("sent_weight_key"):
        sent_weights = sent_w[cfg["sent_weight_key"]]

    train_sampler = None
    if cfg.get("train_sampler") == "joint_balanced":
        train_sampler, summary = build_joint_label_sampler(
            train_df,
            temperature=cfg.get("sampler_temperature", SAMPLER_TEMPERATURE),
            include_neutral_bucket=cfg.get("sampler_include_neutral_bucket", False),
        )
        print("Joint sampler preview:\n", summary.head(10).to_string(index=False))

    trainer = ABSAMultiTaskTrainer(
        model=model, args=args,
        train_dataset=train_encoded, eval_dataset=val_encoded,
        compute_metrics=compute_metrics,
        sent_weights=sent_weights,
        loss_name=cfg["loss_name"],
        use_class_weights=cfg["use_class_weights"],
        focal_gamma=BASE_FOCAL_GAMMA,
        focal_config=cfg.get("focal_config", FOCAL_CONFIG_STANDARD),
        sentiment_loss_weight=cfg.get("sentiment_loss_weight", 0.5),
        aspect_loss_weight=cfg.get("aspect_loss_weight", 0.5),
        scl_weight=cfg.get("scl_weight", 0.0),
        train_sampler=train_sampler,
        presence_weights=pres_w,
        stage2_weights=s2_w,
        aspect_loss_multipliers=aspect_mult,
    )

    print(f"\n=== {cfg['name']} | loss={cfg['loss_name']} | scl={cfg['scl_weight']} ===")
    trainer.train()
    val_metrics = trainer.evaluate(val_encoded)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for ckpt in Path(output_dir).glob("checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)

    return {"output_dir": output_dir, "val_metrics": val_metrics, **cfg}, model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default="./absa_results")
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ── data & weights ────────────────────────────────────────────────────
    train_df, val_df, test_df = load_splits(Path(args.data_root))
    sent_w, pres_w, s2_w     = build_weight_tensors(train_df)
    aspect_mult              = compute_aspect_loss_multipliers(train_df)

    # ── tokenise ──────────────────────────────────────────────────────────
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_encoded  = tokenize_frame(val_df,  tokenizer)
    test_encoded = tokenize_frame(test_df, tokenizer)

    # ── train ─────────────────────────────────────────────────────────────
    seed_everything()
    result, model, tokenizer = run_experiment(
        BEST_EXPERIMENT, train_df, sent_w, pres_w, s2_w, aspect_mult, tokenizer, val_encoded, device)
    print("\nVal metrics:", result["val_metrics"])

    # ── evaluate on test ──────────────────────────────────────────────────
    eval_trainer = ABSAMultiTaskTrainer(
        model=model, args=build_training_args(result["output_dir"] + "/eval_tmp"),
        eval_dataset=test_encoded, compute_metrics=compute_metrics,
        sent_weights=sent_w.get(BEST_EXPERIMENT.get("sent_weight_key")),
        loss_name=BEST_EXPERIMENT["loss_name"],
        use_class_weights=BEST_EXPERIMENT["use_class_weights"],
        focal_gamma=BASE_FOCAL_GAMMA,
        focal_config=BEST_EXPERIMENT.get("focal_config", FOCAL_CONFIG_STANDARD),
        presence_weights=pres_w, stage2_weights=s2_w,
        aspect_loss_multipliers=aspect_mult,
    )
    test_metrics  = eval_trainer.evaluate(test_encoded)
    test_outputs  = eval_trainer.predict(test_encoded)
    val_outputs   = eval_trainer.predict(val_encoded)
    print("Test metrics:", test_metrics)

    # ── calibrate thresholds ──────────────────────────────────────────────
    thresholds, threshold_df = calibrate_presence_thresholds(val_outputs.predictions, val_outputs.label_ids)
    print("\nPresence thresholds:\n", threshold_df.round(4).to_string(index=False))

    test_sent, _, _, test_asps = decode_with_thresholds(test_outputs.predictions, thresholds)
    print("Calibrated test metrics:", metrics_from_decoded(
        test_df["sentiment"].values, test_sent, test_df[ASPECT_COLS].values, test_asps))

    print("\nOverall sentiment report (test):")
    print(classification_report(
        test_df["sentiment"].values, test_sent,
        labels=[0, 1, 2], target_names=["Negative", "Neutral", "Positive"], zero_division=0))

    # ── save ──────────────────────────────────────────────────────────────
    final_dir = Path(args.output_dir) / f"final_{result['name']}"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    threshold_path = final_dir / "presence_thresholds.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model → {final_dir}")
    print(f"Saved thresholds → {threshold_path}")

    # ── demo ──────────────────────────────────────────────────────────────
    for text in ["Sách bị giao trễ 1 tuần, bìa nhàu nát.", "Nội dung hay, đóng gói cẩn thận!"]:
        print(f"  [{text}]  →  {predict_review(text, model, tokenizer, thresholds, device)}")


if __name__ == "__main__":
    main()
