import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("packages/absa_core").resolve()))

import torch
import torch.nn as nn
import numpy as np
import yaml

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

print(">>> TEST 1: Stress-testing pooling layers with edge cases...")
hidden_dim = 768
for name, pooler in [
    ("FirstTokenPooling", FirstTokenPooling()),
    ("MaskedMeanPooling", MaskedMeanPooling(eps=1e-4)),
    ("MultiHeadAttentionPooling", MultiHeadAttentionPooling(hidden_dim, num_heads=4, dropout=0.0)),
]:
    # Edge case: B=1, L=1
    x1 = torch.randn(1, 1, hidden_dim, requires_grad=True)
    m1 = torch.ones(1, 1, dtype=torch.long)
    o1 = pooler(x1, m1)
    assert o1.shape == (1, hidden_dim), f"{name} failed on B=1, L=1"
    o1.sum().backward()
    assert x1.grad is not None and not torch.isnan(x1.grad).any(), f"{name} grad NaN on B=1, L=1"

    # Edge case: All-zero attention mask
    x0 = torch.randn(2, 10, hidden_dim, requires_grad=True)
    m0 = torch.zeros(2, 10, dtype=torch.long)
    o0 = pooler(x0, m0)
    assert o0.shape == (2, hidden_dim), f"{name} failed on all-zero mask"
    assert not torch.isnan(o0).any(), f"{name} output NaN on all-zero mask"
    o0.sum().backward()
    assert x0.grad is not None and not torch.isnan(x0.grad).any(), f"{name} grad NaN on all-zero mask"

    # Edge case: FP16 precision
    pooler_half = pooler.half()
    xh = torch.randn(3, 15, hidden_dim, dtype=torch.float16, requires_grad=True)
    mh = torch.tensor([[1]*15, [1]*5 + [0]*10, [0]*15], dtype=torch.long)
    oh = pooler_half(xh, mh)
    assert oh.shape == (3, hidden_dim), f"{name} failed FP16 shape"
    assert not torch.isnan(oh).any() and not torch.isinf(oh).any(), f"{name} FP16 output NaN/Inf"
    oh.sum().backward()
    assert xh.grad is not None and not torch.isnan(xh.grad).any() and not torch.isinf(xh.grad).any(), f"{name} FP16 grad NaN/Inf"
    pooler.float()

print("Pooling stress tests PASSED.")

print(">>> TEST 2: Hierarchical Head gradient dual-path verification...")
h_head = HierarchicalMultiTaskHead(hidden_size=hidden_dim, os_latent_dim=128, aspect_hidden_dim=384, dropout=0.0)
input_feats = torch.randn(4, hidden_dim, requires_grad=True)
logits = h_head(input_feats)

# Verify that loss from overall sentiment ONLY propagates to os_dense, os_head, AND input_feats
loss_os = logits[0].sum()
loss_os.backward(retain_graph=True)
assert h_head.os_head.weight.grad is not None
assert h_head.os_dense[0].weight.grad is not None
for as_head in h_head.as_heads:
    assert as_head[0].weight.grad is None, "Aspect head should not receive gradient from OS-only loss"
assert input_feats.grad is not None

# Zero grad and verify loss from aspect sentiment ONLY propagates to BOTH base input AND os_dense (via concatenation)
h_head.zero_grad()
input_feats.grad.zero_()
loss_as = sum(l.sum() for l in logits[1:])
loss_as.backward()
for as_head in h_head.as_heads:
    assert as_head[0].weight.grad is not None, "Aspect head must receive gradient from AS loss"
    assert as_head[3].weight.grad is not None
assert h_head.os_dense[0].weight.grad is not None, "OS dense layer must receive gradient from AS loss (dual conditioning)"
assert h_head.os_head.weight.grad is None, "OS head linear classifier should not receive gradient from AS loss"
assert input_feats.grad is not None

print("Hierarchical Head dual-path gradient verification PASSED.")

print(">>> TEST 3: Config files validation...")
for model_yaml in ["ml/configs/models/phobert.yaml", "ml/configs/models/mdeberta.yaml", "ml/configs/models/xlmr.yaml"]:
    cfg = yaml.safe_load(Path(model_yaml).read_text(encoding="utf-8"))
    assert "pooling_type" in cfg, f"Missing pooling_type in {model_yaml}"
    assert "head_type" in cfg, f"Missing head_type in {model_yaml}"
    p = build_pooling_layer(cfg["pooling_type"], 768)
    h = build_task_heads(cfg["head_type"], 768)
    assert p is not None
    assert h is not None
    print(f"  {model_yaml}: pooling={cfg['pooling_type']}, head={cfg['head_type']} -> OK")

print("All Config validations PASSED.")

print(">>> TEST 4: absa_core vs ml parity and deserialization...")
for p_type in ["first_token", "masked_mean", "multihead_attention"]:
    p_ml = build_pooling_layer(p_type, 768, dropout=0.0)
    p_core = core_arch.build_pooling_layer(p_type, 768, dropout=0.0)
    assert set(p_ml.state_dict().keys()) == set(p_core.state_dict().keys())
    p_core.load_state_dict(p_ml.state_dict())

for h_type in ["flat", "hierarchical"]:
    h_ml = build_task_heads(h_type, 768, dropout=0.0)
    h_core = core_arch.build_task_heads(h_type, 768, dropout=0.0)
    assert set(h_ml.state_dict().keys()) == set(h_core.state_dict().keys())
    h_core.load_state_dict(h_ml.state_dict())

print("State dict and architecture parity PASSED.")
print("=== ALL ADVANCED AUDIT TESTS COMPLETED SUCCESSFULLY ===")
