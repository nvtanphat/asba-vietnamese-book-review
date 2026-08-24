from __future__ import annotations

import torch
import torch.nn as nn

TASK_DIMS = [3, 4, 4, 4, 4, 4, 4]


class FirstTokenPooling(nn.Module):
    """Extracts the first token ([CLS] / <s>) representation from transformer encoder outputs."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return hidden_states[:, 0]


class MaskedMeanPooling(nn.Module):
    """Averages hidden states over valid (non-padding) tokens with FP16-safe epsilon clamping."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
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
        B, L, _ = hidden_states.size()

        k = self.key_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.val_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.query.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            mask_bias = (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -10000.0
            scores = scores + mask_bias

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v).squeeze(2).reshape(B, self.hidden_size)
        return self.layer_norm(self.out_proj(context))


def build_pooling_layer(
    pooling_type: str,
    hidden_size: int,
    dropout: float = 0.1,
    num_heads: int = 4,
) -> nn.Module:
    """Factory function to build pooling layers for transformer models."""
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


class FlatMultiTaskHead(nn.Module):
    """Standard parallel independent classification heads."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.15,
        task_dims: list[int] | None = None,
    ):
        super().__init__()
        dims = task_dims or TASK_DIMS
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
        dims = task_dims or TASK_DIMS
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

    Mirrors ml/models/transformer/heads.py's training-time class exactly (same submodule
    names/shapes), since this must load the state_dict of a model trained with head_type
    "two_stage" — `forward()` still returns one [B, num_classes] tensor per task so the rest
    of this predictor (family=="pretrained_encoder" branch) needs no other changes.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.15,
        task_dims: list[int] | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        dims = task_dims or TASK_DIMS
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


class TextCNNNetwork(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        emb = int(config.get("embedding_dim", 200))
        channels = int(config.get("channels", 128))
        kernels = config.get("kernels", [3, 4, 5])
        drop = float(config.get("dropout", 0.35))
        hidden = int(config.get("hidden_dim", 256))
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, channels, int(k)) for k in kernels])
        self.shared = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(channels * len(kernels), hidden),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in TASK_DIMS])

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> list[torch.Tensor]:
        x = self.embedding(input_ids).transpose(1, 2)
        feats = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            xk = torch.nn.functional.pad(x, (0, max(0, k - x.size(-1)))) if x.size(-1) < k else x
            feats.append(torch.relu(conv(xk)).amax(-1))
        h = self.shared(torch.cat(feats, 1))
        return [head(h) for head in self.heads]


class BiLSTMNetwork(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        emb = int(config.get("embedding_dim", 200))
        hidden = int(config.get("hidden_dim", 192))
        layers = int(config.get("num_layers", 2))
        drop = float(config.get("dropout", 0.35))
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(
            emb,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop if layers > 1 else 0,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden * 2, d) for d in TASK_DIMS])

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        x = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        mask = torch.arange(input_ids.size(1), device=input_ids.device)[None, :] < lengths[:, None]
        mean = (out * mask.unsqueeze(-1)).sum(1) / lengths.clamp(min=1).unsqueeze(-1)
        maxv = out.masked_fill(~mask.unsqueeze(-1), -1e9).amax(1)
        h = self.proj(torch.cat([mean, maxv], -1))
        return [head(h) for head in self.heads]


class EncoderMultiTaskNetwork(nn.Module):
    def __init__(
        self,
        config_dir: str,
        dropout: float = 0.15,
        pooling_type: str = "masked_mean",
        head_type: str = "hierarchical",
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.pooling_type = pooling_type
        self.head_type = head_type
        cfg = AutoConfig.from_pretrained(config_dir)
        self.encoder = AutoModel.from_config(cfg)
        hidden = int(cfg.hidden_size)
        self.pooler = build_pooling_layer(pooling_type, hidden, dropout=dropout)
        self.task_head = build_task_heads(head_type, hidden, dropout=dropout, task_dims=TASK_DIMS)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self.pooler(hidden, attention_mask)
        return self.task_head(pooled)
