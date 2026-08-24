# Milestone M1 Architecture & Verification Report: Transformer Feature Pooling (`ml/models/transformer/pooling.py`)

## 1. Observation

### 1.1. Current Architecture Bottlenecks
- **Location**: `ml/models/transformer/model.py:42-47`
```python
def forward(self, input_ids, attention_mask):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state
    # First-token pooling works for RoBERTa/PhoBERT/DeBERTa encoders and keeps architecture identical.
    pooled = self.dropout(hidden[:, 0])
    return [head(pooled) for head in self.heads]
```
- **Evidence**:
  1. `hidden[:, 0]` discards all tokens $t = 1 \dots L-1$, forcing the model to rely solely on the `[CLS]` / `<s>` token representation.
  2. In Vietnamese e-commerce reviews (e.g. Tiki dataset), key aspect descriptors (e.g. *"giao hàng chậm"*, *"đóng gói cẩn thận"*, *"giá hơi cao"*) frequently occur in the middle or final clauses of compound sentences. First-token pooling suffers from representation attenuation across deep layers.
  3. `mdeberta-v3-base` does not use Next Sentence Prediction (NSP) during pretraining and uses disentangled relative attention, making `[CLS]` pooling empirically suboptimal compared to sequence-level aggregation.

### 1.2. Production Parity Requirements
- **Location**: `packages/absa_core/absa_core/models/unified_architectures.py:36-40` and `packages/absa_core/absa_core/models/unified_predictor.py:43`
```python
# unified_architectures.py:
class EncoderMultiTaskNetwork(nn.Module):
    def __init__(self, config_dir: str, dropout: float=0.15):
        super().__init__();from transformers import AutoConfig, AutoModel;cfg=AutoConfig.from_pretrained(config_dir);self.encoder=AutoModel.from_config(cfg);hidden=int(cfg.hidden_size);self.dropout=nn.Dropout(dropout);self.heads=nn.ModuleList([nn.Linear(hidden,d) for d in TASK_DIMS])
    def forward(self,input_ids,attention_mask):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0];h=self.dropout(h);return [head(h) for head in self.heads]
```
- Any architectural change to `EncoderMultiTaskNetwork` in `ml/models/transformer/model.py` must maintain identical module hierarchy and state_dict keys with `packages/absa_core/absa_core/models/unified_architectures.py` to prevent weights-loading failures during production serving.

### 1.3. Existing Test Baseline
- Running `pytest tests/unit tests/smoke` runs 18 tests and passes 100%:
  - `tests/unit/test_calibration.py` (1 passed)
  - `tests/unit/test_dataset_snapshot.py` (1 passed)
  - `tests/unit/test_kaggle_cli_tool.py` (3 passed)
  - `tests/unit/test_losses.py` (3 passed)
  - `tests/unit/test_metrics.py` (2 passed)
  - `tests/unit/test_mlops.py` (4 passed)
  - `tests/unit/test_no_notebooks.py` (1 passed)
  - `tests/unit/test_packaged_maps.py` (1 passed)
  - `tests/unit/test_sklearn_models.py` (1 passed)
  - `tests/smoke/test_registry_smoke.py` (1 passed)

---

## 2. Logic Chain

### 2.1. Mathematical Formulation of Pooling Mechanisms

#### A. FirstTokenPooling (`FirstTokenPooling`)
- **Forward**:
  $$\text{pooled} = H[:, 0, :] \in \mathbb{R}^{B \times D}$$
  where $H \in \mathbb{R}^{B \times L \times D}$ is the last hidden state of the transformer encoder.
- **Parameters**: 0.
- **Role**: Preserves exact backward compatibility with the baseline architecture.

#### B. MaskedMeanPooling (`MaskedMeanPooling`)
- **Forward**:
  Let $M \in \{0, 1\}^{B \times L}$ be the attention mask ($1$ for valid tokens, $0$ for padding tokens).
  Expand mask: $M_{\text{exp}} = M[:, :, \text{None}] \in \mathbb{R}^{B \times L \times 1}$, cast to $H.\text{dtype}$.
  $$\text{sum\_embeddings} = \sum_{t=0}^{L-1} (H[:, t, :] \odot M_{\text{exp}}[:, t, :]) \in \mathbb{R}^{B \times D}$$
  $$\text{sum\_mask} = \text{clamp}\left(\sum_{t=0}^{L-1} M_{\text{exp}}[:, t, :], \min=\epsilon\right) \in \mathbb{R}^{B \times 1}$$
  $$\text{pooled} = \frac{\text{sum\_embeddings}}{\text{sum\_mask}} \in \mathbb{R}^{B \times D}$$
- **Parameters**: 0.
- **Properties**: Uniformly aggregates semantic features across the entire sequence while eliminating padding noise.

#### C. MultiHeadAttentionPooling (`MultiHeadAttentionPooling`)
- **Forward**:
  Let $H_{\text{num}}$ be `num_heads` (default 4), $D_h = D // H_{\text{num}}$ (e.g. $768 // 4 = 192$).
  Learnable query parameters: $Q \in \mathbb{R}^{H_{\text{num}} \times D_h}$, initialized with $\mathcal{N}(0, 0.02^2)$.
  Project Keys and Values:
  $$K = W_k H \in \mathbb{R}^{B \times L \times D} \to [B, H_{\text{num}}, L, D_h]$$
  $$V = W_v H \in \mathbb{R}^{B \times L \times D} \to [B, H_{\text{num}}, L, D_h]$$
  Expand Query:
  $$Q_{\text{exp}} \in \mathbb{R}^{B \times H_{\text{num}} \times 1 \times D_h}$$
  Scaled Dot-Product Attention Scores:
  $$\text{scores} = \frac{Q_{\text{exp}} K^\top}{\sqrt{D_h}} \in \mathbb{R}^{B \times H_{\text{num}} \times 1 \times L}$$
  Mask Penalty (Numerical Stability):
  $$\text{mask\_bias} = (1.0 - M[:, \text{None}, \text{None}, :].\text{to}(\text{scores.dtype})) \times (-10000.0)$$
  $$\text{scores}_{\text{masked}} = \text{scores} + \text{mask\_bias}$$
  $$\text{attn\_weights} = \text{softmax}(\text{scores}_{\text{masked}}, \text{dim}=-1) \in \mathbb{R}^{B \times H_{\text{num}} \times 1 \times L}$$
  $$\text{attn\_weights} = \text{Dropout}(\text{attn\_weights}, p)$$
  $$\text{context} = \text{attn\_weights} \times V \in \mathbb{R}^{B \times H_{\text{num}} \times 1 \times D_h}$$
  Channel Concatenation and Output Projection:
  $$\text{context}_{\text{flat}} = \text{context}.\text{squeeze}(2).\text{reshape}(B, D) \in \mathbb{R}^{B \times D}$$
  $$\text{pooled} = \text{LayerNorm}(W_o \text{context}_{\text{flat}}) \in \mathbb{R}^{B \times D}$$
- **Parameters**: $3 \times D^2 + D \times D_h \cdot H_{\text{num}} + 2D \approx 3D^2 + D \approx 1.77\text{M}$ for $D=768$.

---

### 2.2. Numerical Stability Under FP16 Autocast & Edge Cases

Our empirical testing under pure `torch.float16` and mixed precision revealed two crucial numerical edge-case behaviors:

#### 1. Masked Mean Pooling Clamping Epsilon:
- In `torch.float16`, the maximum representable finite value is $65504.0$. The subnormal resolution limit is $\approx 5.96 \times 10^{-8}$.
- If $\epsilon = 10^{-9}$ is used:
  - In FP16, $10^{-9}$ rounds down to $0.0$.
  - In the backward pass on an all-zero mask, the division gradient $1 / \text{sum\_mask}$ computes $1.0 / 0.0 \to \text{inf}$.
  - $0.0 \times \text{inf} \to \mathbf{NaN}$ in gradient computation!
- If $\epsilon = 10^{-4}$ (or `min=1e-4`):
  - $1.0 / 10^{-4} = 10000.0 < 65504.0$, which is well within FP16 finite range.
  - In the backward pass on an all-zero mask: $0.0 \times 10000.0 = \mathbf{0.0}$ (gradient is perfectly finite and clean).
  - For any valid non-empty sequence, $\text{sum\_mask} \ge 1.0 \gg 10^{-4}$, so $\epsilon = 10^{-4}$ introduces exactly **0.0 distortion** on non-padding tokens.

#### 2. Attention Masking Penalty in Softmax:
- If $-10^9$ or $-\text{inf}$ is used:
  - In FP16, $-10^9$ overflows float16 range and becomes $-\text{inf}$.
  - On an all-padding sequence (or edge sequence), $\text{softmax}([-\text{inf}, -\text{inf}]) \to 0 / 0 \to \mathbf{NaN}$.
- If $-10000.0$ (or $-1e4$) is used:
  - $-10000.0$ is representable in FP16.
  - $\exp(-10000.0) = 0.0$, giving exact zero attention weight to pad tokens without underflow or overflow.
  - On an all-padding sequence, $\text{softmax}([-10000.0, -10000.0]) \to [0.5, 0.5]$ (finite uniform distribution, gradient is finite, no NaN).

#### 3. Attention Head Concatenation Correction:
- The initial survey draft suggested `context.squeeze(2).transpose(1, 2).contiguous().view(B, hidden_size)`.
- *Correction*: Applying `.transpose(1, 2)` to the 3D tensor $[B, H, D_h]$ after `.squeeze(2)` transposed the Head and Dimension axes into $[B, D_h, H]$, interleaving channels $[0, 1, 2, 3, 0, 1, 2, 3]$.
- *Correct implementation*: `context.squeeze(2).reshape(B, self.hidden_size)` correctly preserves sequential head blocks $[0 \dots 191], [192 \dots 383], [384 \dots 575], [576 \dots 767]$.

---

## 3. Caveats

1. **Hardware Discrepancy**:
   - Local workstation execution is CPU-based (`CUDA available? False`).
   - Kaggle remote GPU runs on `NvidiaTeslaT4` with CUDA and AMP FP16.
   - All unit tests must explicitly test both `torch.float32` and `torch.float16` on CPU to guarantee remote Kaggle FP16 stability.
2. **Hidden Size Divisibility**:
   - `MultiHeadAttentionPooling` requires `hidden_size % num_heads == 0`.
   - Default `num_heads = 4` evenly divides `hidden_size = 768` ($192$) for `phobert`, `mdeberta`, and `xlmr`. If larger encoders ($D=1024$) are configured, $1024 \% 4 = 0$ ($256$) also satisfies this.
3. **Optional Attention Mask Handling**:
   - If `attention_mask is None` is passed (e.g. unbatched inference or pre-truncated inputs without padding), `MaskedMeanPooling` must fall back cleanly to `hidden_states.mean(dim=1)` without throwing errors.
   - `FirstTokenPooling` must accept `attention_mask=None` without error.

---

## 4. Conclusion & Concrete Implementation Specifications

### 4.1. Exact Implementation: `ml/models/transformer/pooling.py`

```python
"""Feature pooling strategies for Transformer encoders in SentenAI-Unified."""
from __future__ import annotations

import torch
import torch.nn as nn


class FirstTokenPooling(nn.Module):
    """Extracts the first token ([CLS] / <s>) representation from transformer encoder outputs."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for FirstTokenPooling.

        Args:
            hidden_states: Tensor of shape [B, L, D].
            attention_mask: Optional tensor of shape [B, L] (unused for first token pooling).

        Returns:
            Tensor of shape [B, D].
        """
        return hidden_states[:, 0]


class MaskedMeanPooling(nn.Module):
    """Averages hidden states over valid (non-padding) tokens with FP16-safe epsilon clamping."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for MaskedMeanPooling.

        Args:
            hidden_states: Tensor of shape [B, L, D].
            attention_mask: Optional tensor of shape [B, L] (1 for non-pad, 0 for pad).

        Returns:
            Tensor of shape [B, D].
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # mask_expanded: [B, L, D]
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        # Clamped with eps >= 1e-4 to prevent FP16 gradient overflow (1/eps < 65504) on empty/padded masks
        sum_mask = mask_expanded.sum(dim=1).clamp(min=self.eps)
        return sum_embeddings / sum_mask


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention Pooling over sequence tokens with learnable queries and FP16-safe masking."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.query = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.val_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        nn.init.normal_(self.query, std=0.02)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for MultiHeadAttentionPooling.

        Args:
            hidden_states: Tensor of shape [B, L, D].
            attention_mask: Optional tensor of shape [B, L] (1 for non-pad, 0 for pad).

        Returns:
            Tensor of shape [B, D].
        """
        B, L, _ = hidden_states.size()

        # Project key and value: [B, L, D] -> [B, L, H, D_h] -> [B, H, L, D_h]
        k = self.key_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.val_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand query: [H, D_h] -> [1, H, 1, D_h] -> [B, H, 1, D_h]
        q = self.query.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)

        # Scaled dot-product: [B, H, 1, D_h] @ [B, H, D_h, L] -> [B, H, 1, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            # Mask out padding tokens using -10000.0 for FP16 stability (prevents -inf overflow and NaN)
            mask_bias = (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -10000.0
            scores = scores + mask_bias

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context: [B, H, 1, L] @ [B, H, L, D_h] -> [B, H, 1, D_h] -> [B, D]
        context = torch.matmul(attn_weights, v).squeeze(2).reshape(B, self.hidden_size)
        return self.layer_norm(self.out_proj(context))


def build_pooling_layer(
    pooling_type: str,
    hidden_size: int,
    dropout: float = 0.1,
    num_heads: int = 4
) -> nn.Module:
    """Factory function to build pooling layers for transformer models.

    Args:
        pooling_type: Pooling strategy identifier.
        hidden_size: Encoder hidden dimension size (e.g. 768).
        dropout: Dropout probability for attention weights (if applicable).
        num_heads: Number of attention heads for attention pooling.

    Returns:
        An instantiated nn.Module implementing the pooling forward contract.
    """
    p_type = str(pooling_type).lower().strip().replace("-", "_")
    if p_type in {"cls", "first_token", "first", "firsttoken"}:
        return FirstTokenPooling()
    elif p_type in {"mean", "masked_mean", "average", "avg", "masked_avg", "maskedmean"}:
        return MaskedMeanPooling(eps=1e-4)
    elif p_type in {"attention", "multihead_attention", "mha", "attn", "multi_head_attention", "multiheadattention"}:
        return MultiHeadAttentionPooling(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
    else:
        raise ValueError(
            f"Unsupported pooling_type: '{pooling_type}'. "
            f"Available: 'first_token', 'masked_mean', 'multihead_attention'"
        )
```

---

### 4.2. Exact Implementation: `tests/unit/test_pooling.py`

```python
import pytest
import torch
import torch.nn as nn

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


def test_build_pooling_layer():
    hidden_size = 768

    # First token aliases
    for alias in ["cls", "first_token", "first", "first-token"]:
        p = build_pooling_layer(alias, hidden_size=hidden_size)
        assert isinstance(p, FirstTokenPooling)

    # Masked mean aliases
    for alias in ["mean", "masked_mean", "average", "avg", "masked-mean"]:
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
```

---

### 4.3. Actionable Instructions for the Worker

1. **Step 1 — Create `ml/models/transformer/pooling.py`**:
   - Write `FirstTokenPooling`, `MaskedMeanPooling`, `MultiHeadAttentionPooling`, and `build_pooling_layer` matching Section 4.1.
2. **Step 2 — Create `tests/unit/test_pooling.py`**:
   - Write comprehensive unit tests matching Section 4.2.
   - Run `pytest tests/unit/test_pooling.py -v`.
3. **Step 3 — Run Complete Test Suite**:
   - Run `pytest tests/unit tests/smoke` to ensure all existing and new unit tests pass with zero regressions.

---

## 5. Verification Method

To independently verify the pooling implementation:

```bash
# 1. Run unit test suite for pooling layers
pytest tests/unit/test_pooling.py -v

# 2. Run full unit and smoke test suite
pytest tests/unit tests/smoke

# 3. Direct tensor verification script
python -c "
import torch
from ml.models.transformer.pooling import MaskedMeanPooling, MultiHeadAttentionPooling, FirstTokenPooling

B, L, D = 4, 32, 768
h = torch.randn(B, L, D, dtype=torch.float16, requires_grad=True)
mask = torch.ones(B, L, dtype=torch.long)
mask[:, 20:] = 0

for pooler in [FirstTokenPooling(), MaskedMeanPooling(eps=1e-4).half(), MultiHeadAttentionPooling(D, 4).half()]:
    out = pooler(h, mask)
    assert out.shape == (B, D)
    assert not torch.isnan(out).any()
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(h.grad).any()
    h.grad.zero_()
print('All pooling layers verified under FP16!')
"
```
