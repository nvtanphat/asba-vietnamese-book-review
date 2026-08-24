"""Export the PhoBERT ABSA model ('phobert' variant) to ONNX and quantize to INT8.

CPU cost-optimization pass: converts the deployed checkpoint to ONNX Runtime + INT8
dynamic quantization so the API can serve inference without a full-precision PyTorch
model resident in memory. Only the 'phobert' variant is supported — the BiLSTM variant
uses pack_padded_sequence with data-dependent control flow that doesn't trace cleanly.

Requires the `onnx` extra: uv sync --extra onnx  (from repo root), or
uv pip install -e packages/absa_core[onnx]

Usage (from repo root):
    uv run python packages/absa_core/scripts/export_onnx.py
    uv run python packages/absa_core/scripts/export_onnx.py --out-dir data/models/onnx

Writes <out-dir>/absa_phobert.onnx (fp32) and <out-dir>/absa_phobert.int8.onnx, then
verifies both against the original torch model on a handful of representative
Vietnamese sentences and prints real size/latency numbers — nothing here is asserted
without being measured against this exact checkpoint.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from absa_core.models.architectures import ASPECT_COLS
from absa_core.models.predictor import ABSAPredictor

# A handful of representative Vietnamese reviews spanning all-positive, all-negative,
# mixed, and aspect-free text — enough to catch a broken export, not a statistically
# rigorous accuracy re-validation (that belongs in calibrate_thresholds.py / the eval
# pipeline against a real labeled set).
SAMPLE_TEXTS = [
    "Giao hàng quá chậm, chờ cả tuần. Sách lại bị rách bìa làm mình rất thất vọng.",
    "Nội dung sách cực kỳ hay và ý nghĩa, đóng gói đẹp mắt và bọc chống sốc kỹ càng. Ship nhanh.",
    "Sách in hơi mờ, giá đắt hơn so với các nhà sách khác. Shop tư vấn cũng tạm ổn, shipper cộc lốc.",
    "Bình thường, không có gì đặc biệt.",
    "Tuyệt vời, sẽ ủng hộ shop dài dài!",
]

MAX_LENGTH = 64
N_BENCH_ITERS = 20


class _ExportWrapper(torch.nn.Module):
    """Returns the plain logits tensor — avoids exporting HF's ModelOutput wrapper."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _decode(logits: np.ndarray) -> list[dict]:
    """Mirror absa_service.analyze's decoding, but on a raw logits array."""
    n = len(ASPECT_COLS)
    sent = logits[:, :3]
    pres = logits[:, 3 : 3 + n * 2].reshape(-1, n, 2)
    asp_sent = logits[:, 3 + n * 2 :].reshape(-1, n, 3)

    overall = np.argmax(_softmax(sent), axis=-1)
    pres_probs = _softmax(pres)
    asp_probs = _softmax(asp_sent)

    results = []
    for i in range(logits.shape[0]):
        aspects = {}
        for j, col in enumerate(ASPECT_COLS):
            presence = float(pres_probs[i, j, 1])
            is_present = presence > 0.5  # same fallback threshold as predictor.py sans thresholds.json
            aspects[col] = int(np.argmax(asp_probs[i, j])) if is_present else -1
        results.append({"overall": int(overall[i]), "aspects": aspects})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="hoangloc112/ABSA-TIKI-BOOK")
    ap.add_argument("--out-dir", default="data/models/onnx")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "absa_phobert.onnx"
    int8_path = out_dir / "absa_phobert.int8.onnx"

    print(f"Loading torch model ({args.model_id}, phobert variant)...")
    predictor = ABSAPredictor(model_id=args.model_id, model_variant="phobert")
    torch.set_num_threads(1)  # apples-to-apples with the API's own single-thread-per-worker tuning
    model = predictor.model
    model.eval()

    cleaned = predictor._preprocess(SAMPLE_TEXTS)
    enc = predictor.tokenizer(
        cleaned, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH
    )

    # Capture the reference output *before* tracing/exporting — torch.onnx.export runs
    # the model under a tracer to record its graph, and that appears to leave some
    # attention-implementation selection in a different state afterwards (observed: a
    # huge logit gap when the "reference" was computed post-export vs. pre-export on
    # an otherwise-untouched model). Comparing against a pre-export reference sidesteps
    # that entirely rather than depending on export internals being side-effect-free.
    with torch.no_grad():
        torch_logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.numpy()

    # Same reasoning as above: benchmark the pristine (pre-export) model too, so the
    # "torch" latency number isn't measuring a model whose internals got perturbed by
    # having just been traced.
    def bench_torch() -> float:
        with torch.no_grad():
            for _ in range(3):
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            start = time.perf_counter()
            for _ in range(N_BENCH_ITERS):
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            return (time.perf_counter() - start) / N_BENCH_ITERS * 1000

    torch_ms = bench_torch()

    print(f"Exporting to {fp32_path} ...")
    torch.onnx.export(
        _ExportWrapper(model),
        (enc["input_ids"], enc["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        # Torch >= 2.5 defaults to the dynamo-based exporter, which additionally
        # requires the `onnxscript` package. The legacy TorchScript-based tracer
        # handles this model (plain tensor in, plain tensor out) without it.
        dynamo=False,
    )

    print(f"Quantizing to INT8 -> {int8_path} ...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # ---- verify: same argmax decisions across torch / onnx-fp32 / onnx-int8 ----
    import onnxruntime as ort

    torch_decoded = _decode(torch_logits)

    onnx_inputs = {
        "input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
    }

    def run_onnx_logits(path: Path) -> np.ndarray:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (logits,) = sess.run(["logits"], onnx_inputs)
        return logits

    fp32_logits = run_onnx_logits(fp32_path)
    int8_logits = run_onnx_logits(int8_path)
    fp32_decoded = _decode(fp32_logits)
    int8_decoded = _decode(int8_logits)

    def agreement(a: list[dict], b: list[dict]) -> tuple[int, int]:
        total, matched = 0, 0
        for ra, rb in zip(a, b):
            total += 1
            matched += int(ra["overall"] == rb["overall"])
            for col in ASPECT_COLS:
                total += 1
                matched += int(ra["aspects"][col] == rb["aspects"][col])
        return matched, total

    fp32_match, fp32_total = agreement(torch_decoded, fp32_decoded)
    int8_match, int8_total = agreement(torch_decoded, int8_decoded)
    # This model has no calibrated per-aspect thresholds (thresholds.json doesn't
    # exist yet — see calibrate_thresholds.py), so many presence decisions on these
    # samples sit within a hair of the naive 0.5 fallback cutoff. Reporting the raw
    # logit gap alongside the decision-agreement count distinguishes "the export is
    # numerically faithful, a borderline call flipped" from an actual export bug.
    print(f"\ntorch vs onnx-fp32 max abs logit diff: {np.abs(torch_logits - fp32_logits).max():.2e}")
    print(f"torch vs onnx-fp32 decision agreement:  {fp32_match}/{fp32_total}")
    print(f"torch vs onnx-int8 decision agreement:  {int8_match}/{int8_total}  (quantization trades some precision for the size/speed win below)")

    # ---- size ----
    # In-memory parameter footprint rather than hunting the checkpoint file on disk —
    # the HF cache location varies by machine (HF_HOME), but this is always correct.
    torch_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    fp32_size_mb = fp32_path.stat().st_size / 1e6
    int8_size_mb = int8_path.stat().st_size / 1e6
    print(f"\nModel size — torch (safetensors): {torch_size_mb:.1f} MB")
    print(f"Model size — onnx fp32:            {fp32_size_mb:.1f} MB")
    print(f"Model size — onnx int8:            {int8_size_mb:.1f} MB  ({fp32_size_mb / int8_size_mb:.1f}x smaller than fp32)")

    # ---- latency (single-thread CPU, batch of 5, warmed up); torch_ms measured pre-export above ----
    def bench_onnx(path: Path) -> float:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        for _ in range(3):
            sess.run(["logits"], onnx_inputs)
        start = time.perf_counter()
        for _ in range(N_BENCH_ITERS):
            sess.run(["logits"], onnx_inputs)
        return (time.perf_counter() - start) / N_BENCH_ITERS * 1000

    fp32_ms = bench_onnx(fp32_path)
    int8_ms = bench_onnx(int8_path)
    print(f"\nLatency (batch=5, 1 CPU thread, avg of {N_BENCH_ITERS}) — torch:     {torch_ms:.1f} ms")
    print(f"Latency (batch=5, 1 CPU thread, avg of {N_BENCH_ITERS}) — onnx fp32: {fp32_ms:.1f} ms  ({torch_ms / fp32_ms:.2f}x)")
    print(f"Latency (batch=5, 1 CPU thread, avg of {N_BENCH_ITERS}) — onnx int8: {int8_ms:.1f} ms  ({torch_ms / int8_ms:.2f}x)")

    print(f"\nDone. Point ABSA_ONNX_MODEL_PATH at {int8_path} and set ABSA_USE_ONNX=1 to serve with it.")


if __name__ == "__main__":
    main()
