import pytest
import torch
from ml.models.transformer.heads import (
    FlatMultiTaskHead,
    HierarchicalMultiTaskHead,
    build_task_heads,
)
from ml.models.transformer.pooling import build_pooling_layer
from ml.training.losses import multitask_loss
import absa_core.models.unified_architectures as core_arch


def test_flat_multitask_head_shapes():
    B, D = 4, 768
    head = FlatMultiTaskHead(hidden_size=D, dropout=0.1)
    x = torch.randn(B, D)
    logits = head(x)

    assert isinstance(logits, list)
    assert len(logits) == 7
    assert logits[0].shape == (B, 3)
    for i in range(1, 7):
        assert logits[i].shape == (B, 4)


def test_hierarchical_multitask_head_shapes():
    B, D = 8, 768
    head = HierarchicalMultiTaskHead(hidden_size=D, dropout=0.15, os_latent_dim=128)
    x = torch.randn(B, D)
    logits = head(x)

    assert isinstance(logits, list)
    assert len(logits) == 7
    assert logits[0].shape == (B, 3)
    for i in range(1, 7):
        assert logits[i].shape == (B, 4)


def test_hierarchical_multitask_head_custom_dimensions():
    B, D = 2, 512
    head = HierarchicalMultiTaskHead(
        hidden_size=D,
        dropout=0.1,
        os_latent_dim=64,
        aspect_hidden_dim=128,
    )
    x = torch.randn(B, D)
    logits = head(x)
    assert len(logits) == 7
    assert logits[0].shape == (B, 3)
    for i in range(1, 7):
        assert logits[i].shape == (B, 4)


def test_hierarchical_multitask_head_gradient_flow():
    B, D = 4, 768
    head = HierarchicalMultiTaskHead(hidden_size=D, dropout=0.15)
    x = torch.randn(B, D, requires_grad=True)
    logits = head(x)

    labels = torch.zeros(B, 7, dtype=torch.long)
    labels[:, 0] = torch.randint(0, 3, (B,))
    labels[:, 1:] = torch.randint(0, 4, (B, 6))

    loss = multitask_loss(logits, labels, loss_type="focal")
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

    # Ensure all parameters have valid non-zero gradients
    for name, param in head.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients."


@pytest.mark.parametrize("pooling_type", ["first_token", "masked_mean", "multihead_attention"])
@pytest.mark.parametrize("head_type", ["flat", "hierarchical"])
def test_modular_pooling_and_heads_integration(pooling_type, head_type):
    B, L, D = 4, 16, 64
    hidden = torch.randn(B, L, D, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, 10:] = 0

    pooler = build_pooling_layer(pooling_type, hidden_size=D, num_heads=2)
    head = build_task_heads(head_type, hidden_size=D)

    pooled = pooler(hidden, mask)
    logits = head(pooled)

    assert len(logits) == 7
    assert logits[0].shape == (B, 3)
    for i in range(1, 7):
        assert logits[i].shape == (B, 4)

    labels = torch.zeros(B, 7, dtype=torch.long)
    loss = multitask_loss(logits, labels, loss_type="ce")
    loss.backward()
    assert hidden.grad is not None
    assert torch.all(torch.isfinite(hidden.grad))


def test_build_task_heads_factory():
    D = 768
    flat_head = build_task_heads("flat", hidden_size=D)
    assert isinstance(flat_head, FlatMultiTaskHead)

    linear_head = build_task_heads("linear", hidden_size=D)
    assert isinstance(linear_head, FlatMultiTaskHead)

    hier_head = build_task_heads("hierarchical", hidden_size=D)
    assert isinstance(hier_head, HierarchicalMultiTaskHead)

    cascade_head = build_task_heads(" CASCADE ", hidden_size=D)
    assert isinstance(cascade_head, HierarchicalMultiTaskHead)

    with pytest.raises(ValueError, match="Unsupported head_type"):
        build_task_heads("unknown_head", hidden_size=D)


@pytest.mark.parametrize("batch_size", [1, 2, 16, 32])
def test_hierarchical_head_batch_sizes(batch_size):
    D = 768
    head = HierarchicalMultiTaskHead(hidden_size=D)
    x = torch.randn(batch_size, D)
    logits = head(x)
    assert logits[0].shape == (batch_size, 3)
    for i in range(1, 7):
        assert logits[i].shape == (batch_size, 4)


def test_production_parity_ml_and_absa_core():
    D = 768
    # Test HierarchicalMultiTaskHead parity
    ml_hier = HierarchicalMultiTaskHead(hidden_size=D, dropout=0.0)
    core_hier = core_arch.HierarchicalMultiTaskHead(hidden_size=D, dropout=0.0)

    # State dict keys must be identical
    assert set(ml_hier.state_dict().keys()) == set(core_hier.state_dict().keys())

    # Load ML weights into core model and verify numerical parity
    core_hier.load_state_dict(ml_hier.state_dict())
    ml_hier.eval()
    core_hier.eval()

    x = torch.randn(4, D)
    with torch.no_grad():
        out_ml = ml_hier(x)
        out_core = core_hier(x)

    for o_m, o_c in zip(out_ml, out_core):
        assert torch.allclose(o_m, o_c, atol=1e-6)

    # Test FlatMultiTaskHead parity
    ml_flat = FlatMultiTaskHead(hidden_size=D, dropout=0.0)
    core_flat = core_arch.FlatMultiTaskHead(hidden_size=D, dropout=0.0)
    assert set(ml_flat.state_dict().keys()) == set(core_flat.state_dict().keys())

    core_flat.load_state_dict(ml_flat.state_dict())
    ml_flat.eval()
    core_flat.eval()
    with torch.no_grad():
        out_ml_f = ml_flat(x)
        out_core_f = core_flat(x)
    for o_m, o_c in zip(out_ml_f, out_core_f):
        assert torch.allclose(o_m, o_c, atol=1e-6)
