import pytest
import torch

from ml.models.transformer.pooling import (
    FirstTokenPooling,
    MaskedMeanPooling,
    MultiHeadAttentionPooling,
    build_pooling_layer,
)


def test_first_token_pooling():
    B, L, D = 4, 16, 64
    hidden = torch.randn(B, L, D, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, 10:] = 0

    pooler = FirstTokenPooling()
    out = pooler(hidden, mask)
    assert out.shape == (B, D)
    assert torch.equal(out, hidden[:, 0])

    loss = out.sum()
    loss.backward()
    assert hidden.grad is not None
    # Gradient flows solely to token index 0
    assert torch.count_nonzero(hidden.grad[:, 0, :]) == B * D
    assert torch.count_nonzero(hidden.grad[:, 1:, :]) == 0

    # Also check with attention_mask=None
    out_none = pooler(hidden, None)
    assert torch.equal(out_none, hidden[:, 0])


def test_masked_mean_pooling_accuracy_and_mask_invariance():
    pooler = MaskedMeanPooling(eps=1e-4)
    # 2 samples, 3 tokens, 2 hidden features
    h1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]], requires_grad=True)
    h2 = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [-50.0, 0.0]]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]])

    out1 = pooler(h1, mask)
    out2 = pooler(h2, mask)

    # Changing masked token (index 2) must have 0 effect on output
    assert torch.allclose(out1, out2)
    # Expected mean of [1.0, 2.0] and [3.0, 4.0] is [2.0, 3.0]
    assert torch.allclose(out1, torch.tensor([[2.0, 3.0]]))

    out1.sum().backward()
    # Gradient should be 0.5 on valid tokens and 0.0 on masked token
    expected_grad = torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]])
    assert torch.allclose(h1.grad, expected_grad)

    # Check attention_mask is None fallback
    h_unmasked = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out_unmasked = pooler(h_unmasked, None)
    assert torch.allclose(out_unmasked, torch.tensor([[2.0, 3.0]]))


def test_masked_mean_pooling_fp16_stability():
    pooler = MaskedMeanPooling(eps=1e-4).half()
    h = torch.randn(3, 8, 32, dtype=torch.float16, requires_grad=True)
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],  # Edge case: all padding
    ], dtype=torch.long)

    out = pooler(h, mask)
    assert out.shape == (3, 32)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    loss = out.sum()
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()
    assert not torch.isinf(h.grad).any()


def test_multi_head_attention_pooling_output_and_gradients():
    B, L, D = 4, 20, 128
    pooler = MultiHeadAttentionPooling(hidden_size=D, num_heads=4, dropout=0.0)
    h = torch.randn(B, L, D, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, 15:] = 0

    out = pooler(h, mask)
    assert out.shape == (B, D)
    assert not torch.isnan(out).any()

    loss = out.sum()
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()
    assert pooler.query.grad is not None
    assert pooler.key_proj.weight.grad is not None
    assert pooler.val_proj.weight.grad is not None
    assert pooler.out_proj.weight.grad is not None
    assert pooler.layer_norm.weight.grad is not None

    # Check with attention_mask=None
    out_nomask = pooler(h, None)
    assert out_nomask.shape == (B, D)


def test_multi_head_attention_pooling_fp16_masking_invariance():
    torch.manual_seed(42)
    pooler = MultiHeadAttentionPooling(hidden_size=64, num_heads=2, dropout=0.0).half()
    pooler.eval()

    h1 = torch.randn(2, 6, 64, dtype=torch.float16, requires_grad=True)
    h2 = h1.clone().detach()
    # Heavily corrupt the masked tokens at index 4 and 5
    h2[:, 4:, :] = 999.0

    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.long)

    out1 = pooler(h1, mask)
    out2 = pooler(h2, mask)

    # Corrupted masked tokens must not alter the output
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-4, f"Corrupted tokens altered output by {diff}"

    # Also test backward pass under FP16
    loss = out1.sum()
    loss.backward()
    assert not torch.isnan(h1.grad).any()


def test_multi_head_attention_pooling_all_masked_fp16():
    pooler = MultiHeadAttentionPooling(hidden_size=32, num_heads=2, dropout=0.0).half()
    h = torch.randn(2, 4, 32, dtype=torch.float16, requires_grad=True)
    mask = torch.zeros(2, 4, dtype=torch.long)  # All pad tokens

    out = pooler(h, mask)
    assert out.shape == (2, 32)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    loss = out.sum()
    loss.backward()
    assert not torch.isnan(h.grad).any()


def test_build_pooling_layer():
    hidden_size = 768

    # First token aliases
    for alias in ["cls", "first_token", "first", "first-token", "firsttoken"]:
        p = build_pooling_layer(alias, hidden_size=hidden_size)
        assert isinstance(p, FirstTokenPooling)

    # Masked mean aliases
    for alias in ["mean", "masked_mean", "average", "avg", "masked-mean", "masked_avg"]:
        p = build_pooling_layer(alias, hidden_size=hidden_size)
        assert isinstance(p, MaskedMeanPooling)

    # Attention aliases
    for alias in ["attention", "multihead_attention", "mha", "attn", "multi-head-attention"]:
        p = build_pooling_layer(alias, hidden_size=hidden_size, num_heads=4)
        assert isinstance(p, MultiHeadAttentionPooling)
        assert p.hidden_size == hidden_size
        assert p.num_heads == 4

    # Invalid alias
    with pytest.raises(ValueError, match="Unsupported pooling_type"):
        build_pooling_layer("invalid_pooling", hidden_size=hidden_size)

    # Invalid head dimension
    with pytest.raises(ValueError, match="must be divisible"):
        build_pooling_layer("attention", hidden_size=767, num_heads=4)
