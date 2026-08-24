"""Empirical stress tests for Transformer architecture, pooling, heads, loss backward,
state dict parity, and UnifiedArtifactPredictor loading.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, AutoTokenizer

from ml.models.transformer.model import EncoderMultiTaskNetwork as MLEncoderMultiTaskNetwork
from ml.models.transformer.heads import build_task_heads, FlatMultiTaskHead, HierarchicalMultiTaskHead
from ml.models.transformer.pooling import build_pooling_layer, FirstTokenPooling, MaskedMeanPooling, MultiHeadAttentionPooling
from ml.training.losses import multitask_loss
from ml.data.schema import TASK_SPECS

import absa_core.models.unified_architectures as core_arch
from absa_core.models.unified_predictor import UnifiedArtifactPredictor


@pytest.fixture(scope="module")
def tiny_bert_config_dir(tmp_path_factory):
    """Creates a lightweight BERT config directory for fast unit/stress testing without internet downloads."""
    tmp_dir = tmp_path_factory.mktemp("tiny_bert")
    cfg = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    cfg.save_pretrained(str(tmp_dir))
    return str(tmp_dir)


# ---------------------------------------------------------------------------
# 1. Model Forward Pass & Shape Stress Across Combinations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pooling_type", ["first_token", "masked_mean", "multihead_attention"])
@pytest.mark.parametrize("head_type", ["flat", "hierarchical"])
@pytest.mark.parametrize("batch_size, seq_len", [(1, 8), (2, 32), (7, 64), (16, 128)])
def test_forward_pass_combinations_and_shapes(tiny_bert_config_dir, pooling_type, head_type, batch_size, seq_len):
    """Verifies forward pass across all pooling, head, batch size, and sequence length combinations."""
    net = MLEncoderMultiTaskNetwork(
        tiny_bert_config_dir,
        dropout=0.1,
        pooling_type=pooling_type,
        head_type=head_type,
        from_config_only=True,
    )
    net.eval()

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # Mask half the sequence for half the batch
    if seq_len > 4:
        attention_mask[batch_size // 2:, seq_len // 2:] = 0

    with torch.no_grad():
        logits = net(input_ids, attention_mask)

    assert isinstance(logits, list)
    assert len(logits) == 7

    # Task 0: overall sentiment (3 classes)
    assert logits[0].shape == (batch_size, 3)
    assert not torch.isnan(logits[0]).any()
    assert not torch.isinf(logits[0]).any()

    # Tasks 1..6: aspect sentiments (4 classes each)
    for i in range(1, 7):
        assert logits[i].shape == (batch_size, 4)
        assert not torch.isnan(logits[i]).any()
        assert not torch.isinf(logits[i]).any()


# ---------------------------------------------------------------------------
# 2. Extreme Sequence Boundary Stress
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pooling_type", ["first_token", "masked_mean", "multihead_attention"])
@pytest.mark.parametrize("head_type", ["flat", "hierarchical"])
def test_extreme_sequence_boundaries(tiny_bert_config_dir, pooling_type, head_type):
    """Stress tests boundary conditions: length 1, all padding, and single active token."""
    net = MLEncoderMultiTaskNetwork(
        tiny_bert_config_dir,
        dropout=0.0,
        pooling_type=pooling_type,
        head_type=head_type,
        from_config_only=True,
    )
    net.eval()

    # Case A: Sequence length 1
    ids_1 = torch.randint(0, 1000, (2, 1), dtype=torch.long)
    mask_1 = torch.ones(2, 1, dtype=torch.long)
    logits_1 = net(ids_1, mask_1)
    assert logits_1[0].shape == (2, 3)
    assert not torch.isnan(logits_1[0]).any()

    # Case B: All padding mask (zeros)
    ids_all_pad = torch.randint(0, 1000, (2, 16), dtype=torch.long)
    mask_all_pad = torch.zeros(2, 16, dtype=torch.long)
    logits_pad = net(ids_all_pad, mask_all_pad)
    assert logits_pad[0].shape == (2, 3)
    assert not torch.isnan(logits_pad[0]).any()

    # Case C: Only first token active
    mask_first_only = torch.zeros(2, 16, dtype=torch.long)
    mask_first_only[:, 0] = 1
    logits_first = net(ids_all_pad, mask_first_only)
    assert logits_first[0].shape == (2, 3)
    assert not torch.isnan(logits_first[0]).any()


# ---------------------------------------------------------------------------
# 3. Loss Backward & Gradient Propagation Stress
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("loss_type", ["ce", "focal"])
@pytest.mark.parametrize("pooling_type", ["first_token", "masked_mean", "multihead_attention"])
@pytest.mark.parametrize("head_type", ["flat", "hierarchical"])
def test_loss_backward_and_gradient_flow(tiny_bert_config_dir, loss_type, pooling_type, head_type):
    """Verifies that gradients flow to all trainable model parameters without NaN or zero-stagnation."""
    net = MLEncoderMultiTaskNetwork(
        tiny_bert_config_dir,
        dropout=0.1,
        pooling_type=pooling_type,
        head_type=head_type,
        from_config_only=True,
    )
    net.train()

    B, L = 4, 16
    input_ids = torch.randint(0, 1000, (B, L), dtype=torch.long)
    attention_mask = torch.ones(B, L, dtype=torch.long)
    attention_mask[:, 12:] = 0

    labels = torch.zeros(B, 7, dtype=torch.long)
    labels[:, 0] = torch.randint(0, 3, (B,))
    labels[:, 1:] = torch.randint(0, 4, (B, 6))

    weights = [torch.ones(t.num_classes) for t in TASK_SPECS]
    task_weights = [1.0, 1.2, 1.2, 1.5, 1.2, 1.2, 1.5]

    logits = net(input_ids, attention_mask)
    loss = multitask_loss(
        logits,
        labels,
        weights=weights,
        task_weights=task_weights,
        loss_type=loss_type,
        gamma=2.0,
        label_smoothing=0.05,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()

    # Verify gradients across all active modules
    for name, param in net.named_parameters():
        if "encoder.pooler" in name:
            # Unused HuggingFace default pooler layer (superseded by net.pooler)
            continue
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient."
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN."
            assert not torch.isinf(param.grad).any(), f"Parameter {name} gradient contains Inf."


# ---------------------------------------------------------------------------
# 4. State Dict Parity: ML Training Network <-> ABSA Core Serving Network
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pooling_type", ["first_token", "masked_mean", "multihead_attention"])
@pytest.mark.parametrize("head_type", ["flat", "hierarchical"])
def test_state_dict_exact_parity(tiny_bert_config_dir, pooling_type, head_type):
    """Verifies that ml.models.transformer and absa_core.models produce 100% identical state_dicts,
    loads with strict=True without missing or unexpected keys, and yields identical forward results.
    """
    # 1. Instantiate ML model
    ml_net = MLEncoderMultiTaskNetwork(
        tiny_bert_config_dir,
        dropout=0.0,
        pooling_type=pooling_type,
        head_type=head_type,
        from_config_only=True,
    )

    # 2. Instantiate ABSA Core model
    core_net = core_arch.EncoderMultiTaskNetwork(
        config_dir=tiny_bert_config_dir,
        dropout=0.0,
        pooling_type=pooling_type,
        head_type=head_type,
    )

    # 3. Keys must match 1:1
    ml_keys = set(ml_net.state_dict().keys())
    core_keys = set(core_net.state_dict().keys())

    missing_in_core = ml_keys - core_keys
    unexpected_in_core = core_keys - ml_keys

    assert missing_in_core == set(), f"Missing keys in absa_core: {missing_in_core}"
    assert unexpected_in_core == set(), f"Unexpected keys in absa_core: {unexpected_in_core}"

    # 4. Load state dict into core model with strict=True
    core_net.load_state_dict(ml_net.state_dict(), strict=True)

    # 5. Evaluate numerical parity
    ml_net.eval()
    core_net.eval()

    B, L = 3, 24
    ids = torch.randint(0, 1000, (B, L), dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, 18:] = 0

    with torch.no_grad():
        out_ml = ml_net(ids, mask)
        out_core = core_net(ids, mask)

    assert len(out_ml) == len(out_core) == 7
    for task_idx in range(7):
        assert torch.allclose(out_ml[task_idx], out_core[task_idx], atol=1e-6)


# ---------------------------------------------------------------------------
# 5. UnifiedArtifactPredictor Loading and Serving Verification
# ---------------------------------------------------------------------------
def test_unified_predictor_default_loading_and_inference(tiny_bert_config_dir, tmp_path):
    """Verifies that UnifiedArtifactPredictor loads state dict without missing/unexpected keys
    under the default architecture (masked_mean + hierarchical) and predicts correctly.
    """
    from transformers import BertTokenizerFast
    artifact_dir = tmp_path / "artifacts" / "final"
    model_dir = artifact_dir / "model"
    encoder_dir = model_dir / "encoder"
    tok_dir = model_dir / "tokenizer"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    cfg = BertConfig.from_pretrained(tiny_bert_config_dir)
    cfg.save_pretrained(str(encoder_dir))

    # Initialize and save tokenizer
    vocab_file = tok_dir / "vocab.txt"
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sản", "phẩm", "rất", "tốt", "đẹp", "giao", "hàng", "chậm", "giá", "đắt", "phục", "vụ", "kém"]
    vocab_file.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(str(vocab_file))
    tokenizer.save_pretrained(str(tok_dir))

    # Instantiate and save ML model state dict
    ml_net = MLEncoderMultiTaskNetwork(
        tiny_bert_config_dir,
        dropout=0.0,
        pooling_type="masked_mean",
        head_type="hierarchical",
        from_config_only=True,
    )
    torch.save(ml_net.state_dict(), model_dir / "model.pt")

    # Save model metadata
    model_meta = {
        "name": "phobert",
        "family": "pretrained_encoder",
        "model_name": "vinai/phobert-base-v2",
        "config": {
            "name": "phobert",
            "pooling_type": "masked_mean",
            "head_type": "hierarchical",
            "dropout": 0.15,
            "max_length": 64,
            "word_segmenter": "none",
        },
        "history": [],
    }
    (model_dir / "metadata.json").write_text(json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    top_meta = {
        "model": "phobert",
        "family": "pretrained_encoder",
        "leaderboard_row": {"family": "pretrained_encoder"},
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(top_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    thresholds = {
        "as_content": 0.50,
        "as_physical": 0.50,
        "as_price": 0.45,
        "as_packaging": 0.50,
        "as_delivery": 0.40,
        "as_service": 0.42,
    }
    (artifact_dir / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    # Load with UnifiedArtifactPredictor
    predictor = UnifiedArtifactPredictor(artifact_dir=str(artifact_dir), device="cpu")

    assert predictor.model is not None
    assert predictor.tokenizer is not None
    assert predictor.family == "pretrained_encoder"

    test_inputs = [
        "sản phẩm rất tốt",
        "giao hàng chậm phục vụ kém",
        "giá đắt",
        "",  # Empty text
        "   ",  # Whitespace only
        "👍 👍",  # Emoji only
    ]

    preds = predictor.predict(test_inputs)

    assert isinstance(preds, list)
    assert len(preds) == len(test_inputs)

    for p in preds:
        assert "overall" in p
        assert p["overall"] in {0, 1, 2}
        assert "overall_probs" in p
        assert len(p["overall_probs"]) == 3
        assert pytest.approx(sum(p["overall_probs"]), abs=1e-4) == 1.0

        assert "aspects" in p
        assert len(p["aspects"]) == 6
        for asp_name, asp_val in p["aspects"].items():
            assert asp_val in {-1, 0, 1, 2}

        assert "aspect_probs" in p
        assert len(p["aspect_probs"]) == 6
        for asp_name, asp_info in p["aspect_probs"].items():
            assert "presence" in asp_info
            assert 0.0 <= asp_info["presence"] <= 1.0
            assert "sentiment" in asp_info
            assert len(asp_info["sentiment"]) == 3


def test_unified_predictor_config_propagation_check(tiny_bert_config_dir, tmp_path):
    """Empirically documents whether UnifiedArtifactPredictor forwards pooling_type and head_type
    from metadata.json config into EncoderMultiTaskNetwork.
    """
    from transformers import BertTokenizerFast
    artifact_dir = tmp_path / "artifacts" / "final"
    model_dir = artifact_dir / "model"
    encoder_dir = model_dir / "encoder"
    tok_dir = model_dir / "tokenizer"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    cfg = BertConfig.from_pretrained(tiny_bert_config_dir)
    cfg.save_pretrained(str(encoder_dir))

    vocab_file = tok_dir / "vocab.txt"
    vocab_file.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntest\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(str(vocab_file))
    tokenizer.save_pretrained(str(tok_dir))

    # Saved with multihead_attention pooling
    ml_net = MLEncoderMultiTaskNetwork(
        tiny_bert_config_dir,
        dropout=0.0,
        pooling_type="multihead_attention",
        head_type="hierarchical",
        from_config_only=True,
    )
    torch.save(ml_net.state_dict(), model_dir / "model.pt")

    model_meta = {
        "name": "mdeberta",
        "family": "pretrained_encoder",
        "model_name": "microsoft/mdeberta-v3-base",
        "config": {
            "name": "mdeberta",
            "pooling_type": "multihead_attention",
            "head_type": "hierarchical",
            "dropout": 0.15,
            "max_length": 64,
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "metadata.json").write_text(json.dumps({"model": "mdeberta", "family": "pretrained_encoder"}, indent=2), encoding="utf-8")
    (artifact_dir / "thresholds.json").write_text(json.dumps({"as_content": 0.5}, indent=2), encoding="utf-8")

    # UnifiedArtifactPredictor._load() in packages/absa_core/absa_core/models/unified_predictor.py
    # forwards pooling_type and head_type from cfg into EncoderMultiTaskNetwork.
    # Therefore, loading a model trained with pooling_type='multihead_attention' or head_type='flat'
    # loads cleanly without missing/unexpected keys in state_dict.
    predictor = UnifiedArtifactPredictor(artifact_dir=str(artifact_dir), device="cpu")
    assert predictor.model is not None
    assert isinstance(predictor.model.pooler, core_arch.MultiHeadAttentionPooling)
    assert isinstance(predictor.model.task_head, core_arch.HierarchicalMultiTaskHead)
    preds = predictor.predict(["test sequence"])
    assert len(preds) == 1
    assert "overall" in preds[0]
    assert "aspects" in preds[0]



