
import pytest
import torch
import torch.nn as nn
from ml.models.transformer.pooling import (
    FirstTokenPooling,
    MaskedMeanPooling,
    MultiHeadAttentionPooling,
    build_pooling_layer,
)
from ml.models.transformer.heads import (
    FlatMultiTaskHead,
    HierarchicalMultiTaskHead,
    build_task_heads,
)
from ml.training.losses import multitask_loss
import absa_core.models.unified_architectures as core_arch


# =========================================================================
# 1. EXTREME BOUNDARY CONDITIONS (B=1, L=512, L=1, L=1024, B=64)
# =========================================================================

@pytest.mark.parametrize('pooling_type', ['first_token', 'masked_mean', 'multihead_attention'])
@pytest.mark.parametrize('head_type', ['flat', 'hierarchical'])
@pytest.mark.parametrize('shape', [
    (1, 1, 768),       # Minimal single token, batch size 1
    (1, 512, 768),     # Full max seq length, batch size 1
    (1, 1024, 768),    # Extra long context, batch size 1
    (64, 128, 768),    # Large batch size
    (8, 512, 768),     # Standard batch size with max length 512
])
def test_boundary_shapes_and_forward_backward(pooling_type, head_type, shape):
    B, L, D = shape
    pooler = build_pooling_layer(pooling_type, hidden_size=D, num_heads=4)
    head = build_task_heads(head_type, hidden_size=D)

    h = torch.randn(B, L, D, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    if L > 1:
        mask[:, int(L * 0.8):] = 0

    pooled = pooler(h, mask)
    assert pooled.shape == (B, D)
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()

    logits = head(pooled)
    assert len(logits) == 7
    assert logits[0].shape == (B, 3)
    for i in range(1, 7):
        assert logits[i].shape == (B, 4)

    # Multi-task loss backward pass
    targets = torch.zeros(B, 7, dtype=torch.long)
    loss = multitask_loss(logits, targets, loss_type='focal')
    loss.backward()

    assert h.grad is not None
    assert torch.all(torch.isfinite(h.grad))
    assert not torch.isnan(h.grad).any()


# =========================================================================
# 2. EXTREME MASK CONFIGURATIONS (ALL-PADDING, SINGLE-TOKEN, SPARSE)
# =========================================================================

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('pooling_type', ['first_token', 'masked_mean', 'multihead_attention'])
def test_extreme_mask_scenarios(dtype, pooling_type):
    B, L, D = 5, 64, 768
    pooler = build_pooling_layer(pooling_type, hidden_size=D, num_heads=4).to(dtype)
    head = build_task_heads('hierarchical', hidden_size=D).to(dtype)

    h = torch.randn(B, L, D, dtype=dtype, requires_grad=True)
    
    # 5 extreme mask rows:
    # 0: all zeros (all-padding)
    # 1: single token at index 0
    # 2: single token at last index (L-1)
    # 3: single token in the middle
    # 4: all ones (no padding)
    mask = torch.zeros(B, L, dtype=torch.long)
    mask[1, 0] = 1
    mask[2, -1] = 1
    mask[3, L // 2] = 1
    mask[4, :] = 1

    pooled = pooler(h, mask)
    assert pooled.shape == (B, D)
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()

    logits = head(pooled)
    for logit in logits:
        assert not torch.isnan(logit).any()
        assert not torch.isinf(logit).any()

    # Test gradient backprop
    loss = sum(l.float().sum() for l in logits)
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()
    assert not torch.isinf(h.grad).any()


# =========================================================================
# 3. PAD IMMUNITY & MASK INVARIANCE ADVERSARIAL ATTACK
# =========================================================================

def test_masked_mean_pad_corruption_invariance():
    pooler = MaskedMeanPooling(eps=1e-4)
    B, L, D = 4, 32, 128
    
    h_clean = torch.randn(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.long)
    mask[:, :10] = 1

    h_corrupted = h_clean.clone()
    h_corrupted[:, 10:, :] = 1e5

    out_clean = pooler(h_clean, mask)
    out_corrupted = pooler(h_corrupted, mask)

    assert torch.allclose(out_clean, out_corrupted, atol=1e-5)


def test_multihead_attention_pad_corruption_invariance_fp16():
    pooler = MultiHeadAttentionPooling(hidden_size=64, num_heads=2, dropout=0.0).half().eval()
    B, L, D = 2, 16, 64
    
    h_clean = torch.randn(B, L, D, dtype=torch.float16)
    mask = torch.zeros(B, L, dtype=torch.long)
    mask[:, :4] = 1

    h_corrupted = h_clean.clone()
    h_corrupted[:, 4:, :] = 500.0

    out_clean = pooler(h_clean, mask)
    out_corrupted = pooler(h_corrupted, mask)

    diff = (out_clean - out_corrupted).abs().max().item()
    assert diff < 1e-3, f'Padded token corruption leaked into output with max diff: {diff}'


# =========================================================================
# 4. FP16 AUTOCAST & NUMERICAL DYNAMICS
# =========================================================================

@pytest.mark.parametrize('pooling_type', ['first_token', 'masked_mean', 'multihead_attention'])
def test_fp16_gradient_flow_and_numerical_safety(pooling_type):
    B, L, D = 4, 64, 768
    pooler = build_pooling_layer(pooling_type, hidden_size=D).half()
    head = build_task_heads('hierarchical', hidden_size=D).half()

    h = torch.randn(B, L, D, dtype=torch.float16, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, 30:] = 0

    pooled = pooler(h, mask)
    logits = head(pooled)

    targets = torch.zeros(B, 7, dtype=torch.long)
    loss = multitask_loss(logits, targets, loss_type='ce')
    loss.backward()

    assert h.grad is not None
    assert not torch.isnan(h.grad).any()
    assert not torch.isinf(h.grad).any()


# =========================================================================
# 5. DROPOUT BEHAVIOR (TRAIN VS EVAL DETERMINISM)
# =========================================================================

def test_heads_and_pooling_eval_determinism():
    D = 768
    head = HierarchicalMultiTaskHead(hidden_size=D, dropout=0.5)
    pooler = MultiHeadAttentionPooling(hidden_size=D, num_heads=4, dropout=0.5)
    
    head.eval()
    pooler.eval()

    x = torch.randn(4, 32, D)
    mask = torch.ones(4, 32, dtype=torch.long)

    p1 = pooler(x, mask)
    p2 = pooler(x, mask)
    assert torch.equal(p1, p2)

    l1 = head(p1)
    l2 = head(p1)
    for out1, out2 in zip(l1, l2):
        assert torch.equal(out1, out2)


def test_heads_and_pooling_train_stochasticity():
    torch.manual_seed(123)
    D = 768
    head = HierarchicalMultiTaskHead(hidden_size=D, dropout=0.5)
    pooler = MultiHeadAttentionPooling(hidden_size=D, num_heads=4, dropout=0.5)
    
    head.train()
    pooler.train()

    x = torch.randn(4, 32, D)
    mask = torch.ones(4, 32, dtype=torch.long)

    p1 = pooler(x, mask)
    p2 = pooler(x, mask)
    assert not torch.equal(p1, p2)


# =========================================================================
# 6. GRADIENT SPARSITY FOR PADDED TOKENS
# =========================================================================

def test_masked_mean_gradient_sparsity():
    B, L, D = 2, 8, 16
    pooler = MaskedMeanPooling(eps=1e-4)
    h = torch.randn(B, L, D, requires_grad=True)
    mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
    ], dtype=torch.long)

    out = pooler(h, mask)
    loss = out.sum()
    loss.backward()

    assert h.grad is not None
    assert torch.all(h.grad[0, :3, :] != 0.0)
    assert torch.all(h.grad[0, 3:, :] == 0.0)
    assert torch.all(h.grad[1, :5, :] != 0.0)
    assert torch.all(h.grad[1, 5:, :] == 0.0)
