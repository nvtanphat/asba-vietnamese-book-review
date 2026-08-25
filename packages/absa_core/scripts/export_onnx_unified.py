"""Export the promoted (champion) unified pretrained-encoder model to ONNX Runtime.

Unlike scripts/export_onnx.py (which targets a different, legacy HF-hub-hosted PhoBERT
checkpoint with a flat presence+sentiment head), this reads whatever model
`UnifiedArtifactPredictor` would actually load from a promoted artifact directory —
currently PhoBERT with a two_stage aspect head — straight off `<artifact-dir>/model`, so
it stays correct for any future promoted pretrained_encoder champion (phobert/xlmr/mdeberta).

Requires the `onnx` extra: uv pip install -e "packages/absa_core[onnx]"

Usage (from repo root):
    uv run python packages/absa_core/scripts/export_onnx_unified.py
    uv run python packages/absa_core/scripts/export_onnx_unified.py --artifact-dir artifacts/final --out-dir artifacts/final/onnx

Writes <out-dir>/model.onnx (fp32), model.fp16.onnx and model.int8.onnx, then verifies all
three against the original torch model (numerically and on final present/absent + sentiment
decisions, using the artifact's own calibrated thresholds.json) and benchmarks real
single-request (batch=1), single-CPU-thread latency for all four — nothing here is asserted
without being measured against this exact checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

TASK_DIMS = [3, 4, 4, 4, 4, 4, 4]
ASPECT_COLS = ["as_content", "as_physical", "as_price", "as_packaging", "as_delivery", "as_service"]

# A handful of representative Vietnamese reviews spanning all-positive, all-negative,
# mixed, and aspect-free text — enough to catch a broken export/quantization, not a
# statistically rigorous accuracy re-validation (that belongs against the sealed test split).
SAMPLE_TEXTS = [
    "Giao hàng quá chậm, chờ cả tuần. Sách lại bị rách bìa làm mình rất thất vọng.",
    "Nội dung sách cực kỳ hay và ý nghĩa, đóng gói đẹp mắt và bọc chống sốc kỹ càng. Ship nhanh.",
    "Sách in hơi mờ, giá đắt hơn so với các nhà sách khác. Shop tư vấn cũng tạm ổn, shipper cộc lốc.",
    "Bình thường, không có gì đặc biệt.",
    "Tuyệt vời, sẽ ủng hộ shop dài dài!",
]
N_BENCH_ITERS = 100
LATENCY_TARGET_MS = 20.0


class _ExportWrapper(torch.nn.Module):
    """Concatenates the two_stage head's per-task logits into one [B, 27] tensor — ONNX
    graphs need a fixed output signature, not the variable-length Python list the model
    normally returns."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.cat(outs, dim=-1)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _split(flat: np.ndarray) -> list[np.ndarray]:
    """[B, 27] -> 7 arrays matching TASK_DIMS (overall + 6 aspects)."""
    out, offset = [], 0
    for d in TASK_DIMS:
        out.append(flat[:, offset : offset + d])
        offset += d
    return out


def _decode(flat: np.ndarray, thresholds: dict) -> list[dict]:
    """Mirror UnifiedArtifactPredictor.predict()'s decoding exactly: each aspect tensor is
    already the two_stage head's combined [neg,neu,pos,absent] distribution, so presence is
    `1 - p[absent]` compared against the artifact's own calibrated per-aspect threshold —
    not a naive 0.5 cutoff."""
    parts = _split(flat)
    overall = np.argmax(_softmax(parts[0]), axis=-1)
    results = []
    for i in range(flat.shape[0]):
        aspects: dict[str, int] = {}
        for j, col in enumerate(ASPECT_COLS):
            p = _softmax(parts[j + 1][i : i + 1])[0]
            presence = 1.0 - p[3]
            present = presence >= float(thresholds.get(col, 0.5))
            aspects[col] = int(np.argmax(p[:3])) if present else -1
        results.append({"overall": int(overall[i]), "aspects": aspects})
    return results


def _agreement(reference: list[dict], other: list[dict]) -> tuple[int, int]:
    total = matched = 0
    for ra, rb in zip(reference, other):
        total += 1
        matched += int(ra["overall"] == rb["overall"])
        for col in ASPECT_COLS:
            total += 1
            matched += int(ra["aspects"][col] == rb["aspects"][col])
    return matched, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default="artifacts/final")
    ap.add_argument("--model-dir", default=None, help="Override: directory with model.pt/encoder/tokenizer/metadata.json directly inside (default: <artifact-dir>/model)")
    ap.add_argument("--thresholds", default=None, help="Override: path to a thresholds.json (default: <artifact-dir>/thresholds.json)")
    ap.add_argument("--out-dir", default=None, help="Default: <artifact-dir>/onnx")
    ap.add_argument("--max-length", type=int, default=None, help="Default: the artifact's own config.max_length")
    args = ap.parse_args()

    root = Path(args.artifact_dir)
    model_dir = Path(args.model_dir) if args.model_dir else root / "model"
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"No promoted model metadata at {meta_path}. Promote a champion first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = meta.get("config", {})
    family = meta.get("family")
    if family != "pretrained_encoder":
        raise SystemExit(
            f"ONNX export only supports family=='pretrained_encoder' champions (got {family!r}). "
            "TextCNN/BiLSTM use pack_padded_sequence control flow that doesn't trace cleanly; "
            "classical (sklearn) models don't need ONNX at all."
        )

    thresholds_path = Path(args.thresholds) if args.thresholds else root / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8")) if thresholds_path.exists() else {}

    out_dir = Path(args.out_dir) if args.out_dir else root / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model.onnx"
    fp16_path = out_dir / "model.fp16.onnx"
    int8_path = out_dir / "model.int8.onnx"

    print(f"Loading champion model ({meta.get('name')}, head_type={cfg.get('head_type')}) from {model_dir} ...")
    from transformers import AutoTokenizer

    from absa_core.models.unified_architectures import EncoderMultiTaskNetwork

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    net = EncoderMultiTaskNetwork(
        str(model_dir / "encoder"),
        float(cfg.get("dropout", 0.15)),
        pooling_type=str(cfg.get("pooling_type", "masked_mean")),
        head_type=str(cfg.get("head_type", "hierarchical")),
    )
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    net.eval()
    torch.set_num_threads(1)  # apples-to-apples with the API's own single-thread-per-worker tuning

    max_length = args.max_length or int(cfg.get("max_length", 160))
    batch_enc = tokenizer(SAMPLE_TEXTS, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    # Single-request (batch=1) is what the API actually serves per call — the <20ms target
    # is a per-request number for a real (short) review, not a padded-to-160-tokens number.
    # Forcing padding="max_length" here would benchmark the worst case a review can ever
    # be, not what real traffic looks like (measured 3-4x slower than natural length).
    single_enc = tokenizer(SAMPLE_TEXTS[0], return_tensors="pt", padding=False, truncation=True, max_length=max_length)

    with torch.no_grad():
        torch_flat = torch.cat(net(batch_enc["input_ids"], batch_enc["attention_mask"]), dim=-1).numpy()

    def bench_torch() -> tuple[float, float]:
        with torch.no_grad():
            for _ in range(20):
                net(single_enc["input_ids"], single_enc["attention_mask"])
            samples = []
            for _ in range(N_BENCH_ITERS):
                start = time.perf_counter()
                net(single_enc["input_ids"], single_enc["attention_mask"])
                samples.append((time.perf_counter() - start) * 1000)
        samples.sort()
        return samples[len(samples) // 2], samples[int(len(samples) * 0.9)]

    torch_ms, torch_p90_ms = bench_torch()

    print(f"Exporting fp32 -> {fp32_path}")
    torch.onnx.export(
        _ExportWrapper(net),
        (batch_enc["input_ids"], batch_enc["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        # Torch >= 2.5 defaults to the dynamo-based exporter, which additionally requires
        # the `onnxscript` package. The legacy TorchScript-based tracer handles this model
        # (plain tensors in, one plain tensor out) without it.
        dynamo=False,
    )

    print(f"Converting fp16 -> {fp16_path}")
    # The generic onnxconverter_common.float16.convert_float_to_float16 leaves this
    # encoder's attention-mask Cast/Div nodes in an inconsistent type state (ONNX Runtime
    # then refuses to load the graph at all) — onnxruntime.transformers.optimizer is the
    # tool actually built for BERT-family attention graphs and produces a loadable model.
    from onnxruntime.transformers import optimizer as ort_transformers_optimizer

    opt_model = ort_transformers_optimizer.optimize_model(str(fp32_path), model_type="bert", use_gpu=False)
    opt_model.convert_float_to_float16(keep_io_types=True)
    opt_model.save_model_to_file(str(fp16_path))

    print(f"Quantizing int8 -> {int8_path}")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    import onnxruntime as ort

    def run(path: Path) -> tuple[np.ndarray, float, float]:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        batch_inputs = {"input_ids": batch_enc["input_ids"].numpy(), "attention_mask": batch_enc["attention_mask"].numpy()}
        (batch_logits,) = sess.run(["logits"], batch_inputs)
        single_inputs = {"input_ids": single_enc["input_ids"].numpy(), "attention_mask": single_enc["attention_mask"].numpy()}
        for _ in range(20):
            sess.run(["logits"], single_inputs)
        samples = []
        for _ in range(N_BENCH_ITERS):
            start = time.perf_counter()
            sess.run(["logits"], single_inputs)
            samples.append((time.perf_counter() - start) * 1000)
        samples.sort()
        median_ms = samples[len(samples) // 2]
        p90_ms = samples[int(len(samples) * 0.9)]
        return batch_logits, median_ms, p90_ms

    fp32_logits, fp32_ms, fp32_p90 = run(fp32_path)
    fp16_logits, fp16_ms, fp16_p90 = run(fp16_path)
    int8_logits, int8_ms, int8_p90 = run(int8_path)

    torch_decoded = _decode(torch_flat, thresholds)
    print(f"\ntorch vs onnx-fp32 max abs logit diff: {np.abs(torch_flat - fp32_logits).max():.2e}")
    for name, logits in [("fp32", fp32_logits), ("fp16", fp16_logits), ("int8", int8_logits)]:
        matched, total = _agreement(torch_decoded, _decode(logits, thresholds))
        print(f"torch vs onnx-{name} decision agreement: {matched}/{total}")

    def size_mb(path: Path) -> float:
        return path.stat().st_size / 1e6

    torch_mb = sum(p.numel() * p.element_size() for p in net.parameters()) / 1e6
    print(f"\nModel size — torch: {torch_mb:.1f} MB")
    print(f"Model size — onnx fp32: {size_mb(fp32_path):.1f} MB")
    print(f"Model size — onnx fp16: {size_mb(fp16_path):.1f} MB")
    print(f"Model size — onnx int8: {size_mb(int8_path):.1f} MB")

    print(f"\nLatency — single request (batch=1, natural length, no forced padding), 1 CPU thread, median/p90 of {N_BENCH_ITERS} runs:")
    print(f"  torch:     median {torch_ms:6.2f} ms  p90 {torch_p90_ms:6.2f} ms")
    for name, ms, p90 in [("onnx fp32", fp32_ms, fp32_p90), ("onnx fp16", fp16_ms, fp16_p90), ("onnx int8", int8_ms, int8_p90)]:
        verdict = "OK <20ms" if p90 < LATENCY_TARGET_MS else ("median OK, p90 borderline" if ms < LATENCY_TARGET_MS else "does NOT meet target")
        print(f"  {name}: median {ms:6.2f} ms  p90 {p90:6.2f} ms  ({torch_ms / ms:.2f}x vs torch median)  [{verdict}]")
    if fp16_ms > fp32_ms:
        print(
            "\nNote: onnx fp16 is SLOWER than fp32 here — ONNX Runtime's CPU execution provider has no "
            "native FP16 compute kernels for most transformer ops on this hardware; it casts to fp32, "
            "computes, casts back. FP16 is a GPU (tensor-core) optimization, not a CPU one — use int8 "
            "for CPU serving."
        )

    print(f"\nWrote: {fp32_path}, {fp16_path}, {int8_path}")
    print("Point ABSA_ONNX_MODEL_PATH at the fastest one that still agrees closely with torch above, and set ABSA_USE_ONNX=1.")


if __name__ == "__main__":
    main()
