"""Hierarchical and Flat Task Heads for ABSA."""
from __future__ import annotations

import torch
import torch.nn as nn
from ml.data.schema import TASK_SPECS


class FlatMultiTaskHead(nn.Module):
    """Standard parallel independent classification heads."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.15,
        task_dims: list[int] | None = None,
    ):
        super().__init__()
        dims = task_dims or [t.num_classes for t in TASK_SPECS]
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, d) for d in dims])

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h = self.dropout(pooled)
        return [head(h) for head in self.heads]


class HierarchicalMultiTaskHead(nn.Module):
    """Hierarchical head conditioning aspect sentiment classifications on overall sentiment latent features."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.15,
        os_latent_dim: int = 128,
        aspect_hidden_dim: int | None = None,
        task_dims: list[int] | None = None,
    ):
        super().__init__()
        dims = task_dims or [t.num_classes for t in TASK_SPECS]
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.os_latent_dim = os_latent_dim
        self.aspect_hidden_dim = aspect_hidden_dim or (hidden_size // 2)

        # Overall sentiment branch (3 classes: neg=0, neu=1, pos=2)
        self.os_dense = nn.Sequential(
            nn.Linear(hidden_size, os_latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.os_head = nn.Linear(os_latent_dim, dims[0])

        # Aspect sentiment branches (6 aspects, 4 classes each: neg=0, neu=1, pos=2, abs=3)
        combined_dim = hidden_size + os_latent_dim
        self.as_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, self.aspect_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.aspect_hidden_dim, d),
            )
            for d in dims[1:]
        ])

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h_base = self.dropout(pooled)
        h_os = self.os_dense(h_base)
        logits_os = self.os_head(h_os)

        # Condition aspect heads on concatenation of base and overall sentiment latent
        h_combined = torch.cat([h_base, h_os], dim=-1)
        logits_aspects = [head(h_combined) for head in self.as_heads]

        return [logits_os, *logits_aspects]


class PresenceSentimentMultiTaskHead(nn.Module):
    """Two-stage aspect heads: for each aspect, a presence head (absent vs. present) and a
    separate sentiment head (negative/neutral/positive) instead of one shared 4-class head.
    Ports the decomposition from the specialized legacy PhoBERT trainer (ml/legacy_phobert),
    which reached f1_combined=0.795 on this dataset vs. ~0.65 for the unified single-head
    models — decoupling presence from polarity keeps the dominant "absent" label from
    drowning out the sentiment signal for sparse aspects (price, service).

    `forward()` still returns one [B, num_classes] tensor per TASK_SPECS entry (num_classes=4
    for aspects) so the rest of the pipeline (predict_proba, calibration, evaluation) needs no
    changes: presence/sentiment probabilities are combined into a proper joint 4-class
    distribution and log()'d back into "logits" (softmax(log(p)) == p). The raw per-aspect
    presence/sentiment logits used to actually train each stage with its own loss/weights are
    cached on `last_presence_logits` / `last_sentiment_logits` after every forward call.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.15,
        task_dims: list[int] | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        dims = task_dims or [t.num_classes for t in TASK_SPECS]
        n_aspects = len(dims) - 1
        self.eps = float(eps)
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, dims[0])
        self.presence_heads = nn.ModuleList([nn.Linear(hidden_size, 2) for _ in range(n_aspects)])
        self.aspect_sentiment_heads = nn.ModuleList([nn.Linear(hidden_size, 3) for _ in range(n_aspects)])
        self.last_presence_logits: list[torch.Tensor] = []
        self.last_sentiment_logits: list[torch.Tensor] = []

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h = self.dropout(pooled)
        overall_logits = self.sentiment_head(h)

        self.last_presence_logits = [head(h) for head in self.presence_heads]
        self.last_sentiment_logits = [head(h) for head in self.aspect_sentiment_heads]

        combined = []
        for presence_logits, sentiment_logits in zip(self.last_presence_logits, self.last_sentiment_logits):
            presence_probs = torch.softmax(presence_logits, dim=-1)
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
            joint_present = presence_probs[:, 1:2] * sentiment_probs
            joint_absent = presence_probs[:, 0:1]
            joint = torch.cat([joint_present, joint_absent], dim=-1)
            combined.append(torch.log(joint.clamp(min=self.eps)))

        return [overall_logits, *combined]


def build_task_heads(
    head_type: str,
    hidden_size: int,
    dropout: float = 0.15,
    os_latent_dim: int = 128,
    aspect_hidden_dim: int | None = None,
    task_dims: list[int] | None = None,
) -> nn.Module:
    """Factory function for instantiating multi-task classification heads."""
    h_type = str(head_type).lower().strip().replace("-", "_")
    if h_type in {"flat", "linear", "independent"}:
        return FlatMultiTaskHead(hidden_size, dropout=dropout, task_dims=task_dims)
    elif h_type in {"hierarchical", "cascade", "hier"}:
        return HierarchicalMultiTaskHead(
            hidden_size=hidden_size,
            dropout=dropout,
            os_latent_dim=os_latent_dim,
            aspect_hidden_dim=aspect_hidden_dim,
            task_dims=task_dims,
        )
    elif h_type in {"two_stage", "presence_sentiment", "presence_polarity"}:
        return PresenceSentimentMultiTaskHead(hidden_size, dropout=dropout, task_dims=task_dims)
    else:
        raise ValueError(
            f"Unsupported head_type: '{head_type}'. Available: 'flat', 'hierarchical', 'two_stage'"
        )
