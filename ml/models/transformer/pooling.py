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
    num_heads: int = 4,
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
