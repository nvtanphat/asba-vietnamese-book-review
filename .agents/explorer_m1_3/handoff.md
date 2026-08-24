# Milestone M1 Integration & Production Sync Handoff Report

**Target Milestone**: M1 (Transformer Architecture & Pooling Optimization)  
**Agent**: `explorer_m1_3` (Integration, Model Checkpointing Lifecycle & Production Serving Parity)  
**Status**: Ready for Worker Implementation  

---

## 1. Observation

### 1.1. Current Transformer Implementation in Training Pipeline
In `ml/models/transformer/model.py:26-48`:
```python
class EncoderMultiTaskNetwork(nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.15, from_config_only: bool = False):
        super().__init__()
        if from_config_only:
            cfg = AutoConfig.from_pretrained(model_name_or_path)
            self.encoder = AutoModel.from_config(cfg, torch_dtype=torch.float32)
        else:
            self.encoder = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
        self.encoder = self.encoder.float()
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, t.num_classes) for t in TASK_SPECS])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        return [head(pooled) for head in self.heads]
```
`TransformerMultiTaskABSA` instantiates `EncoderMultiTaskNetwork` in `ml/models/transformer/model.py:63`:
```python
self.model = EncoderMultiTaskNetwork(self.model_name, float(config.get("dropout", 0.15))).to(self.device)
```

### 1.2. Production Mirror in `packages/absa_core`
In `packages/absa_core/absa_core/models/unified_architectures.py:5, 36-40`:
```python
TASK_DIMS = [3, 4, 4, 4, 4, 4, 4]

class EncoderMultiTaskNetwork(nn.Module):
    def __init__(self, config_dir: str, dropout: float=0.15):
        super().__init__();from transformers import AutoConfig, AutoModel;cfg=AutoConfig.from_pretrained(config_dir);self.encoder=AutoModel.from_config(cfg);hidden=int(cfg.hidden_size);self.dropout=nn.Dropout(dropout);self.heads=nn.ModuleList([nn.Linear(hidden,d) for d in TASK_DIMS])
    def forward(self,input_ids,attention_mask):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0];h=self.dropout(h);return [head(h) for head in self.heads]
```
In `packages/absa_core/absa_core/models/unified_predictor.py:41-43`:
```python
elif self.family=="pretrained_encoder":
    from transformers import AutoTokenizer
    cfg=self.model_meta.get("config",{})
    self.tokenizer=AutoTokenizer.from_pretrained(self.model_dir/"tokenizer")
    net=EncoderMultiTaskNetwork(str(self.model_dir/"encoder"),float(cfg.get("dropout",0.15)))
    net.load_state_dict(torch.load(self.model_dir/"model.pt",map_location=self.device))
    self.model=net.to(self.device).eval()
    self.max_length=int(cfg.get("max_length",160))
    self.word_segmenter=cfg.get("word_segmenter","none")
```

### 1.3. Model Checkpointing & Serialization Lifecycle
In `ml/models/transformer/model.py:108-150`:
- **Periodic Checkpoint (`last.pt`)**:
  ```python
  if last: torch.save({"model":self.model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"best":best,"bad":bad},last)
  ```
- **Validation Champion (`best.pt`)**:
  ```python
  if score>best:
      best,bad=score,0
      if out: torch.save(self.model.state_dict(),out/"best.pt")
  ```
- **Final Model Artifact (`model.pt`, `encoder/`, `tokenizer/`, `metadata.json`)**:
  ```python
  def save(self,output_dir):
      out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
      (out/"encoder").mkdir(exist_ok=True); self.model.encoder.config.save_pretrained(out/"encoder"); self.tokenizer.save_pretrained(out/"tokenizer")
      torch.save(self.model.state_dict(),out/"model.pt")
      (out/"metadata.json").write_text(json.dumps({"name":self.name,"family":self.family,"model_name":self.model_name,"config":self.config,"history":self.history},ensure_ascii=False,indent=2),encoding="utf-8")
  ```

### 1.4. Current Model Configuration YAML Files
In `ml/configs/models/phobert.yaml`, `mdeberta.yaml`, `xlmr.yaml`:
```yaml
model_name: vinai/phobert-base-v2  # or microsoft/mdeberta-v3-base or FacebookAI/xlm-roberta-base
max_length: 160
batch_size: 16
epochs: 6
patience: 2
lr: 0.00002
weight_decay: 0.01
warmup_ratio: 0.10
dropout: 0.15
label_smoothing: 0.05
max_class_weight: 6.0
max_grad_norm: 1.0
fp16: true
loss_type: focal
focal_gamma: 2.0
```

### 1.5. Remote Kaggle & Production Promotion Handshake
- `tools/kaggle_cli/cli.py:360-368` exports `model.pt`, `last.pt`, `best.pt`, `encoder/`, `tokenizer/`, `metadata.json`, `metrics.json` to `/kaggle/working/sentenai-output/<MODEL>`.
- `tools/kaggle_cli/cli.py:544-547` syncs the exported artifact back to local `experiments/<MODEL>`.
- `ml/benchmark.py:22-39` promotes the best validation model by copying `experiments/<MODEL>` to `artifacts/final/model` and writing `artifacts/final/metadata.json` and `artifacts/final/thresholds.json`.
- `apps/api/app/services/absa_service.py:36-39` loads the promoted artifact with `UnifiedArtifactPredictor("artifacts/final")`.

---

## 2. Logic Chain

### 2.1. Structural Alignment for Zero-Downtime State Dict Loading
1. **Observation**: `UnifiedArtifactPredictor` loads weights directly using `net.load_state_dict(torch.load(self.model_dir / "model.pt"))`.
2. **Mechanism**: PyTorch `nn.Module.load_state_dict()` performs strict key matching against the module's registered parameter and buffer names.
3. **Requirement**: If `ml/models/transformer/model.py` changes `self.heads` to `self.pooler` and `self.task_head`, then `packages/absa_core/absa_core/models/unified_architectures.py` MUST mirror the exact attribute names:
   - `self.encoder` (`AutoModel`)
   - `self.pooler` (`FirstTokenPooling`, `MaskedMeanPooling`, or `MultiHeadAttentionPooling`)
   - `self.task_head` (`FlatMultiTaskHead` or `HierarchicalMultiTaskHead`)
4. **Hierarchical Head Parameter Keys**:
   - `task_head.os_dense.0.weight`, `task_head.os_dense.0.bias`
   - `task_head.os_head.weight`, `task_head.os_head.bias`
   - `task_head.as_heads.0.0.weight`, `task_head.as_heads.0.0.bias`, `task_head.as_heads.0.3.weight`, `task_head.as_heads.0.3.bias`
   - ... through `task_head.as_heads.5.3.weight`, `task_head.as_heads.5.3.bias`.
5. **Deduction**: Synchronizing the exact module hierarchy between training and serving guarantees 100% parameter name congruence and zero weight loading errors.

### 2.2. Dynamic Architecture Deserialization via Metadata
1. **Observation**: `TransformerMultiTaskABSA.save()` dumps `self.config` into `metadata.json`.
2. **Mechanism**: When `ml/configs/models/*.yaml` includes `pooling_type: masked_mean` and `head_type: hierarchical`, this configuration is saved into `experiments/<model>/metadata.json` and copied to `artifacts/final/model/metadata.json`.
3. **Deduction**: `UnifiedArtifactPredictor._load()` can reliably extract `pooling_type = cfg.get("pooling_type", "masked_mean")` and `head_type = cfg.get("head_type", "hierarchical")`, ensuring that the correct architecture is instantiated at inference time without hardcoded assumptions.

### 2.3. Output Contract Preservation
1. **Observation**: Downstream consumers (`multitask_loss`, `_predict_loader`, `calibrate_absent_thresholds`, `decode_probabilities`, `UnifiedArtifactPredictor.predict`) expect `model(input_ids, attention_mask)` to return a list of 7 tensors:
   - Index 0: Overall Sentiment logits, shape `(B, 3)`
   - Indices 1..6: Aspect Sentiment logits, shape `(B, 4)` for `as_content`, `as_physical`, `as_price`, `as_packaging`, `as_delivery`, `as_service`.
2. **Deduction**: `HierarchicalMultiTaskHead.forward()` returns `[logits_os, *logits_aspects]` where `logits_os` is $(B, 3)$ and each aspect logit is $(B, 4)$. The output signature `list[Tensor]` of length 7 is 100% preserved. No changes are required in `ml/training/losses.py`, `ml/evaluation/`, or `ml/train.py`.

---

## 3. Caveats

1. **Standalone Requirement of `packages/absa_core`**:
   - `packages/absa_core` is a separate package installed in isolated production environments where `ml/` is not on `sys.path`.
   - `packages/absa_core` must NOT import from `ml.*`.
   - All pooling classes (`FirstTokenPooling`, `MaskedMeanPooling`, `MultiHeadAttentionPooling`, `build_pooling_layer`) and head classes (`FlatMultiTaskHead`, `HierarchicalMultiTaskHead`, `build_task_heads`) must be self-contained in `packages/absa_core/absa_core/models/unified_architectures.py`.
2. **Config File Dual Locations**:
   - The workspace contains both `ml/configs/models/*.yaml` (used by `ml.train`) and `ml/configs/experiments/*.yaml` (used by some experiment scripts). Both locations must be updated to keep hyperparameters synchronized.
3. **Word Segmenter Setting**:
   - `phobert.yaml` specifies `word_segmenter: pyvi`. `mdeberta.yaml` and `xlmr.yaml` do not use `pyvi`. The new architecture retains this per-model tokenizer preprocessing setting cleanly in `_normalize_model_input` and `UnifiedArtifactPredictor._clean`.

---

## 4. Conclusion & Concrete Worker Instructions

### 4.1. Step 1: Create `ml/models/transformer/pooling.py`
Create `ml/models/transformer/pooling.py` with:
```python
"""Feature pooling strategies for Transformer encoders."""
from __future__ import annotations

import torch
import torch.nn as nn


class FirstTokenPooling(nn.Module):
    """Extracts first token ([CLS] / <s>) representation."""
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return hidden_states[:, 0]


class MaskedMeanPooling(nn.Module):
    """Averages hidden states over valid (non-padding) tokens with numerical stability clamp."""
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention Pooling over sequence tokens with learnable queries and FP16-safe masking."""
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
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

        context = torch.matmul(attn_weights, v).squeeze(2).transpose(1, 2).contiguous().view(B, self.hidden_size)
        return self.layer_norm(self.out_proj(context))


def build_pooling_layer(pooling_type: str, hidden_size: int, dropout: float = 0.1) -> nn.Module:
    p_type = str(pooling_type).lower().strip()
    if p_type in {"cls", "first_token", "first"}:
        return FirstTokenPooling()
    elif p_type in {"mean", "masked_mean", "average"}:
        return MaskedMeanPooling()
    elif p_type in {"attention", "multihead_attention", "mha"}:
        return MultiHeadAttentionPooling(hidden_size=hidden_size, num_heads=4, dropout=dropout)
    else:
        raise ValueError(f"Unsupported pooling_type: '{pooling_type}'. Available: 'first_token', 'masked_mean', 'multihead_attention'")
```

### 4.2. Step 2: Create `ml/models/transformer/heads.py`
Create `ml/models/transformer/heads.py` with:
```python
"""Hierarchical and Flat Multi-Task Classification Heads for ABSA."""
from __future__ import annotations

import torch
import torch.nn as nn
from ml.data.schema import TASK_SPECS


class FlatMultiTaskHead(nn.Module):
    """Standard parallel classification heads from pooled representations."""
    def __init__(self, hidden_size: int, dropout: float = 0.15, task_dims: list[int] | None = None):
        super().__init__()
        dims = task_dims or [t.num_classes for t in TASK_SPECS]
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, d) for d in dims])

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h = self.dropout(pooled)
        return [head(h) for head in self.heads]


class HierarchicalMultiTaskHead(nn.Module):
    """Hierarchical head conditioning 6 aspect sentiment branches on overall sentiment latent features."""
    def __init__(self, hidden_size: int, dropout: float = 0.15, os_latent_dim: int = 128, task_dims: list[int] | None = None):
        super().__init__()
        dims = task_dims or [t.num_classes for t in TASK_SPECS]
        self.dropout = nn.Dropout(dropout)
        
        # Overall Sentiment branch (3 classes: neg, neu, pos)
        self.os_dense = nn.Sequential(
            nn.Linear(hidden_size, os_latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.os_head = nn.Linear(os_latent_dim, dims[0])
        
        # Aspect Sentiment branches (6 aspects, 4 classes each)
        combined_dim = hidden_size + os_latent_dim
        self.as_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, d)
            ) for d in dims[1:]
        ])

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h_base = self.dropout(pooled)
        h_os = self.os_dense(h_base)
        logits_os = self.os_head(h_os)
        
        h_combined = torch.cat([h_base, h_os], dim=-1)
        logits_aspects = [head(h_combined) for head in self.as_heads]
        return [logits_os, *logits_aspects]


def build_task_heads(head_type: str, hidden_size: int, dropout: float = 0.15, task_dims: list[int] | None = None) -> nn.Module:
    h_type = str(head_type).lower().strip()
    if h_type in {"flat", "linear", "independent"}:
        return FlatMultiTaskHead(hidden_size, dropout=dropout, task_dims=task_dims)
    elif h_type in {"hierarchical", "cascade", "hier"}:
        return HierarchicalMultiTaskHead(hidden_size, dropout=dropout, task_dims=task_dims)
    else:
        raise ValueError(f"Unsupported head_type: '{head_type}'. Available: 'flat', 'hierarchical'")
```

### 4.3. Step 3: Update `ml/models/transformer/model.py`
Update `EncoderMultiTaskNetwork` and `TransformerMultiTaskABSA.__init__`:
```python
from ml.models.transformer.pooling import build_pooling_layer
from ml.models.transformer.heads import build_task_heads

class EncoderMultiTaskNetwork(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        dropout: float = 0.15,
        pooling_type: str = "masked_mean",
        head_type: str = "hierarchical",
        from_config_only: bool = False,
    ):
        super().__init__()
        self.pooling_type = pooling_type
        self.head_type = head_type
        if from_config_only:
            cfg = AutoConfig.from_pretrained(model_name_or_path)
            self.encoder = AutoModel.from_config(cfg, torch_dtype=torch.float32)
        else:
            self.encoder = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
        self.encoder = self.encoder.float()
        hidden = int(self.encoder.config.hidden_size)
        self.pooler = build_pooling_layer(pooling_type, hidden, dropout=dropout)
        self.task_head = build_task_heads(head_type, hidden, dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self.pooler(hidden, attention_mask)
        return self.task_head(pooled)
```
In `TransformerMultiTaskABSA.__init__`:
```python
        dropout = float(config.get("dropout", 0.15))
        pooling_type = str(config.get("pooling_type", "masked_mean"))
        head_type = str(config.get("head_type", "hierarchical"))
        self.model = EncoderMultiTaskNetwork(
            self.model_name,
            dropout=dropout,
            pooling_type=pooling_type,
            head_type=head_type,
        ).to(self.device)
```

### 4.4. Step 4: Synchronize `packages/absa_core/absa_core/models/unified_architectures.py`
In `packages/absa_core/absa_core/models/unified_architectures.py`, add self-contained poolers and heads, and update `EncoderMultiTaskNetwork`:
```python
from __future__ import annotations
import torch
import torch.nn as nn

TASK_DIMS = [3, 4, 4, 4, 4, 4, 4]


class FirstTokenPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return hidden_states[:, 0]


class MaskedMeanPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
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
        context = torch.matmul(attn_weights, v).squeeze(2).transpose(1, 2).contiguous().view(B, self.hidden_size)
        return self.layer_norm(self.out_proj(context))


def build_pooling_layer(pooling_type: str, hidden_size: int, dropout: float = 0.1) -> nn.Module:
    p_type = str(pooling_type).lower().strip()
    if p_type in {"cls", "first_token", "first"}:
        return FirstTokenPooling()
    elif p_type in {"mean", "masked_mean", "average"}:
        return MaskedMeanPooling()
    elif p_type in {"attention", "multihead_attention", "mha"}:
        return MultiHeadAttentionPooling(hidden_size=hidden_size, num_heads=4, dropout=dropout)
    else:
        raise ValueError(f"Unsupported pooling_type: '{pooling_type}'")


class FlatMultiTaskHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.15, task_dims: list[int] | None = None):
        super().__init__()
        dims = task_dims or TASK_DIMS
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, d) for d in dims])

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h = self.dropout(pooled)
        return [head(h) for head in self.heads]


class HierarchicalMultiTaskHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.15, os_latent_dim: int = 128, task_dims: list[int] | None = None):
        super().__init__()
        dims = task_dims or TASK_DIMS
        self.dropout = nn.Dropout(dropout)
        self.os_dense = nn.Sequential(
            nn.Linear(hidden_size, os_latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.os_head = nn.Linear(os_latent_dim, dims[0])
        combined_dim = hidden_size + os_latent_dim
        self.as_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, d)
            ) for d in dims[1:]
        ])

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h_base = self.dropout(pooled)
        h_os = self.os_dense(h_base)
        logits_os = self.os_head(h_os)
        h_combined = torch.cat([h_base, h_os], dim=-1)
        logits_aspects = [head(h_combined) for head in self.as_heads]
        return [logits_os, *logits_aspects]


def build_task_heads(head_type: str, hidden_size: int, dropout: float = 0.15, task_dims: list[int] | None = None) -> nn.Module:
    h_type = str(head_type).lower().strip()
    if h_type in {"flat", "linear", "independent"}:
        return FlatMultiTaskHead(hidden_size, dropout=dropout, task_dims=task_dims)
    elif h_type in {"hierarchical", "cascade", "hier"}:
        return HierarchicalMultiTaskHead(hidden_size, dropout=dropout, task_dims=task_dims)
    else:
        raise ValueError(f"Unsupported head_type: '{head_type}'")


class TextCNNNetwork(nn.Module): ... # unchanged
class BiLSTMNetwork(nn.Module): ...  # unchanged


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
        cfg = AutoConfig.from_pretrained(config_dir)
        self.encoder = AutoModel.from_config(cfg)
        hidden = int(cfg.hidden_size)
        self.pooler = build_pooling_layer(pooling_type, hidden, dropout=dropout)
        self.task_head = build_task_heads(head_type, hidden, dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self.pooler(hidden, attention_mask)
        return self.task_head(pooled)
```

### 4.5. Step 5: Update `packages/absa_core/absa_core/models/unified_predictor.py`
In `packages/absa_core/absa_core/models/unified_predictor.py:41-45`:
```python
        elif self.family == "pretrained_encoder":
            from transformers import AutoTokenizer
            cfg = self.model_meta.get("config", {})
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "tokenizer")
            net = EncoderMultiTaskNetwork(
                str(self.model_dir / "encoder"),
                dropout=float(cfg.get("dropout", 0.15)),
                pooling_type=str(cfg.get("pooling_type", "masked_mean")),
                head_type=str(cfg.get("head_type", "hierarchical")),
            )
            net.load_state_dict(torch.load(self.model_dir / "model.pt", map_location=self.device))
            self.model = net.to(self.device).eval()
            self.max_length = int(cfg.get("max_length", 160))
            self.word_segmenter = cfg.get("word_segmenter", "none")
```

### 4.6. Step 6: Update YAML Config Files
In `ml/configs/models/phobert.yaml`, `mdeberta.yaml`, `xlmr.yaml` (and mirrors in `ml/configs/experiments/`):
- Add `pooling_type: masked_mean`
- Add `head_type: hierarchical`

### 4.7. Step 7: Create Comprehensive Unit Test `tests/unit/test_transformer_architectures.py`
Create `tests/unit/test_transformer_architectures.py` testing:
1. `MaskedMeanPooling`, `MultiHeadAttentionPooling`, `FirstTokenPooling` forward passes & stability with zero-padding masks.
2. `HierarchicalMultiTaskHead` and `FlatMultiTaskHead` shape verification ($B \times 3$ for OS, $B \times 4$ for 6 AS branches).
3. Backward pass & gradient flow through all heads and pooler parameters.
4. Cross-compatibility and `load_state_dict` equivalence between `ml.models.transformer.model.EncoderMultiTaskNetwork` and `absa_core.models.unified_architectures.EncoderMultiTaskNetwork`.

---

## 5. Verification Method

### 5.1. Independent Verification Script
Run the following PowerShell / python command to verify architecture shapes, state dict synchronization, and gradient backprop:
```powershell
python -c "
import torch
import torch.nn as nn
from ml.models.transformer.pooling import MaskedMeanPooling, MultiHeadAttentionPooling, FirstTokenPooling, build_pooling_layer
from ml.models.transformer.heads import HierarchicalMultiTaskHead, FlatMultiTaskHead, build_task_heads
from ml.training.losses import multitask_loss

B, L, D = 4, 32, 768
hidden = torch.randn(B, L, D, requires_grad=True)
mask = torch.ones(B, L, dtype=torch.long)
mask[:, 20:] = 0  # 12 pad tokens

# 1. Poolers
for p_name in ['first_token', 'masked_mean', 'multihead_attention']:
    pooler = build_pooling_layer(p_name, D)
    p_out = pooler(hidden, mask)
    assert p_out.shape == (B, D), f'{p_name} output shape mismatch: {p_out.shape}'

# 2. Heads
labels = torch.randint(0, 3, (B, 7))
for h_name in ['flat', 'hierarchical']:
    head = build_task_heads(h_name, D)
    logits = head(p_out)
    assert len(logits) == 7, f'{h_name} head count mismatch: {len(logits)}'
    assert logits[0].shape == (B, 3), f'{h_name} OS shape mismatch: {logits[0].shape}'
    for i in range(1, 7):
        assert logits[i].shape == (B, 4), f'{h_name} AS shape mismatch: {logits[i].shape}'
    loss = multitask_loss(logits, labels)
    loss.backward()
    assert hidden.grad is not None

print('=== ALL INTEGRATION CHECKS PASSED ===')
"
```

### 5.2. Running Full Test Suite
```powershell
pytest tests/unit
```
Ensure all tests pass with 0 failures.

### 5.3. Invalidation Conditions
- Any parameter name mismatch between `ml.models.transformer` and `absa_core.models.unified_architectures` that causes `load_state_dict` `Missing key(s)` or `Unexpected key(s)` errors.
- Any output dimension divergence from $(B, 3)$ for OS or $(B, 4)$ for AS.
- Missing `pooling_type` or `head_type` in `ml/configs/models/*.yaml`.
