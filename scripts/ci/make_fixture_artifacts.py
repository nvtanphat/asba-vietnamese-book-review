"""Build tiny, randomly-initialized model weights for CI's Docker-build smoke test.

`artifacts/final/` and `artifacts/ensemble/` track their config/tokenizer/metadata JSON
files in git, but the actual `.pt`/`.onnx` weights are gitignored (hundreds of MB — see
docs/kaggle_cli.md and the model registry for how a real promotion gets those). A fresh
checkout is therefore structurally complete but missing weight files, and
`docker build -f infra/api.Dockerfile .` fails on the `COPY artifacts/...` step without
them.

This script does NOT reproduce the real champion — it builds a same-shaped (same
vocab_size/pad_token_id, so token ids from the *real* tracked tokenizers stay valid;
tiny hidden_size/layers, so it runs in seconds) network from the real tracked HF config
and tokenizer, purely so the Docker build (and, if ever added, a container-boot smoke
test) exercises the real code path — `EncoderMultiTaskNetwork`, `UnifiedArtifactPredictor`,
`EnsembleOnnxPredictor` — without needing model weights or Kaggle credentials in CI.

Usage (from repo root): python scripts/ci/make_fixture_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoConfig, AutoTokenizer

TASK_DIMS = [3, 4, 4, 4, 4, 4, 4]


def _tiny_config(real_config_path: Path) -> AutoConfig:
    """A same-vocab, same-special-tokens RobertaConfig, shrunk for CI speed."""
    real = json.loads(real_config_path.read_text(encoding="utf-8"))
    cfg = AutoConfig.for_model(
        real.get("model_type", "roberta"),
        vocab_size=real["vocab_size"],
        pad_token_id=real.get("pad_token_id", 1),
        bos_token_id=real.get("bos_token_id", 0),
        eos_token_id=real.get("eos_token_id", 2),
        type_vocab_size=real.get("type_vocab_size", 1),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=real.get("max_position_embeddings", 130),
    )
    return cfg


def build_member(*, real_config_path: Path, tokenizer_dir: Path, out_config_dir: Path, out_model_pt: Path, out_onnx: Path | None) -> None:
    from absa_core.models.unified_architectures import EncoderMultiTaskNetwork

    out_config_dir.mkdir(parents=True, exist_ok=True)
    tiny_cfg = _tiny_config(real_config_path)
    tiny_cfg.save_pretrained(out_config_dir)

    net = EncoderMultiTaskNetwork(str(out_config_dir), 0.1, pooling_type="masked_mean", head_type="two_stage")
    net.eval()
    out_model_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_model_pt)
    print(f"Wrote {out_model_pt} ({sum(p.numel() for p in net.parameters())} params)")

    if out_onnx is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        enc = tokenizer(["fixture text for CI"], return_tensors="pt", padding=True, truncation=True, max_length=16)

        class _Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                return torch.cat(self.m(input_ids=input_ids, attention_mask=attention_mask), dim=-1)

        out_onnx.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            _Wrapper(net),
            (enc["input_ids"], enc["attention_mask"]),
            str(out_onnx),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        print(f"Wrote {out_onnx}")


def main() -> None:
    root = _PROJECT_ROOT

    # artifacts/final: torch weights for UnifiedArtifactPredictor's direct (non-ONNX) path.
    build_member(
        real_config_path=root / "artifacts/final/model/encoder/config.json",
        tokenizer_dir=root / "artifacts/final/model/tokenizer",
        out_config_dir=root / "artifacts/final/model/encoder",
        out_model_pt=root / "artifacts/final/model/model.pt",
        out_onnx=None,
    )

    # artifacts/ensemble: each member only needs an ONNX file (EnsembleOnnxPredictor never
    # builds the torch network) — reuse artifacts/final's real tracked config as the source
    # shape for both slots since CI is validating the build/serving *code path*, not model
    # identity.
    for member in ("phobert", "xlmr"):
        member_dir = root / "artifacts/ensemble" / member
        if not member_dir.exists():
            continue
        build_member(
            real_config_path=root / "artifacts/final/model/encoder/config.json",
            tokenizer_dir=member_dir / "tokenizer",
            out_config_dir=root / f".ci_tmp/{member}_encoder",
            out_model_pt=root / f".ci_tmp/{member}_model.pt",
            out_onnx=member_dir / "model.int8.onnx",
        )


if __name__ == "__main__":
    main()
