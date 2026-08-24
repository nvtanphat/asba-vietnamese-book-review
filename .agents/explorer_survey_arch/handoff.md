# Architecture Survey & Design Report: Transformer Models, Feature Pooling, and Hierarchical Heads in SentenAI-Unified

## 1. Observation

### 1.1. Model Registry and File Locations
In `ml/models/registry.py:4-13` and `ml/models/registry.py:16-20`:
```python
MODEL_REGISTRY = {
    "logistic": "ml.models.logistic.model",
    "linear_svm": "ml.models.linear_svm.model",
    "textcnn": "ml.models.textcnn.model",
    "bilstm": "ml.models.bilstm.model",
    "phobert": "ml.models.transformer.model",
    "xlmr": "ml.models.transformer.model",
    "mdeberta": "ml.models.transformer.model",
    "vit5": "ml.models.vit5.model",
}

def build_model(name: str, config: dict):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    module = importlib.import_module(MODEL_REGISTRY[name])
    return module.build({**config, "name": name})
```

All three pretrained Transformer encoders (`phobert`, `xlmr`, `mdeberta`) share the exact same implementation in `ml/models/transformer/model.py`.

Pretrained checkpoint mappings in `ml/models/transformer/model.py:19-23`:
```python
MODEL_NAMES = {
    "phobert": "vinai/phobert-base-v2",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
}
```

### 1.2. Base Model Interface Contract
In `ml/models/base.py:8-24`:
```python
class ABSABenchmarkModel(ABC):
    name: str
    family: str

    @abstractmethod
    def fit(self, train_texts, train_y: np.ndarray, val_texts=None, val_y=None, *, output_dir: str | Path | None = None, resume: bool = False): ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> list[np.ndarray]:
        """Return seven probability arrays: overall (N,3), six aspect arrays (N,4)."""

    @abstractmethod
    def save(self, output_dir: str | Path): ...

    def parameter_count(self) -> int | None:
        return None
```

### 1.3. Task Schema and Label Definitions
In `ml/data/schema.py:5-28`:
```python
ASPECT_COLS = [
    "as_content",
    "as_physical",
    "as_price",
    "as_packaging",
    "as_delivery",
    "as_service",
]
TARGET_COLS = ["sentiment", *ASPECT_COLS]
SENTIMENT_CLASSES = (0, 1, 2)
ASPECT_CLASSES = (0, 1, 2, 3)
ABSENT_CLASS = 3
SENTIMENT_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
ASPECT_SENTIMENT_NAMES = {0: "negative", 1: "neutral", 2: "positive", 3: "absent"}

@dataclass(frozen=True)
class TaskSpec:
    name: str
    num_classes: int

TASK_SPECS = [TaskSpec("sentiment", 3), *[TaskSpec(c, 4) for c in ASPECT_COLS]]
```

### 1.4. Current Network Architecture and First-Token Pooling
In `ml/models/transformer/model.py:26-48`:
```python
class EncoderMultiTaskNetwork(nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.15, from_config_only: bool = False):
        super().__init__()
        # Force fp32 weights regardless of the checkpoint's declared torch_dtype: GradScaler
        # requires fp32 master parameters and raises "Attempting to unscale FP16 gradients"
        # if a HF config (e.g. some mdeberta-v3 mirrors) causes the encoder to load in fp16.
        if from_config_only:
            cfg = AutoConfig.from_pretrained(model_name_or_path)
            self.encoder = AutoModel.from_config(cfg, torch_dtype=torch.float32)
        else:
            self.encoder = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
        self.encoder = self.encoder.float()  # belt-and-suspenders: guarantee fp32 params/buffers
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, t.num_classes) for t in TASK_SPECS])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        # First-token pooling works for RoBERTa/PhoBERT/DeBERTa encoders and keeps architecture identical.
        pooled = self.dropout(hidden[:, 0])
        return [head(pooled) for head in self.heads]
```

Production mirror in `packages/absa_core/absa_core/models/unified_architectures.py:36-40`:
```python
class EncoderMultiTaskNetwork(nn.Module):
    def __init__(self, config_dir: str, dropout: float=0.15):
        super().__init__();from transformers import AutoConfig, AutoModel;cfg=AutoConfig.from_pretrained(config_dir);self.encoder=AutoModel.from_config(cfg);hidden=int(cfg.hidden_size);self.dropout=nn.Dropout(dropout);self.heads=nn.ModuleList([nn.Linear(hidden,d) for d in TASK_DIMS])
    def forward(self,input_ids,attention_mask):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0];h=self.dropout(h);return [head(h) for head in self.heads]
```

### 1.5. Forward Pass, Multitask Loss, and Inference Interface
In `ml/training/losses.py:16-29`:
```python
def multitask_loss(logits, labels, weights=None, task_weights=None, loss_type: str = "ce", gamma: float = 2.0, label_smoothing: float = 0.0):
    """Average or task-weighted per-task loss across a multitask head list. `logits` is a list of [B, C_i]
    tensors (one per task), `labels` is a [B, T] tensor with one column per task."""
    weights = weights or [None] * len(logits)
    if loss_type == "focal":
        per_task = [focal_loss(x, labels[:, i], weight=weights[i], gamma=gamma, label_smoothing=label_smoothing) for i, x in enumerate(logits)]
    else:
        per_task = [F.cross_entropy(x, labels[:, i], weight=weights[i], label_smoothing=label_smoothing) for i, x in enumerate(logits)]
    if task_weights is not None:
        tw = torch.as_tensor(task_weights, dtype=torch.float32, device=logits[0].device)
        stacked = torch.stack(per_task)
        return (stacked * tw).sum() / tw.sum()
    return sum(per_task) / len(logits)
```

In `ml/models/transformer/model.py:86-93`:
```python
def _predict_loader(self, loader):
    probs = [[] for _ in TASK_SPECS]; self.model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = self.model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
            for i, x in enumerate(logits): probs[i].append(torch.softmax(x, -1).cpu().numpy())
    return [np.concatenate(p, 0) if p else np.empty((0, t.num_classes)) for p, t in zip(probs, TASK_SPECS)]
```

In `ml/models/transformer/model.py:114-126` (Training step):
```python
for step_idx, batch in enumerate(train_loader):
    labels = batch.pop("labels").to(self.device); ids = batch["input_ids"].to(self.device); mask = batch["attention_mask"].to(self.device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
        logits = self.model(ids, mask)
        loss = multitask_loss(logits, labels, weights=weights, task_weights=task_weights, loss_type=self.config.get("loss_type","ce"), gamma=float(self.config.get("focal_gamma",2.0)), label_smoothing=float(self.config.get("label_smoothing",0.05)))
        loss_step = loss / accum_steps
    scaler.scale(loss_step).backward()
```

### 1.6. Configuration Files
Current YAML configuration in `ml/configs/models/phobert.yaml`, `mdeberta.yaml`, `xlmr.yaml`:
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

---

## 2. Logic Chain

### 2.1. Feature Pooling Bottleneck & Justification for Upgrades
- **Observation**: `ml/models/transformer/model.py:46` currently extracts `hidden[:, 0]`, discarding token states $t=1 \dots L-1$.
- **Problem**: In Vietnamese e-commerce reviews (e.g. Tiki book reviews), reviews are frequently compound sentences with multiple clauses where aspect sentiment terms appear in the middle or end of the sequence (e.g., *"Sách nội dung rất hay, in ấn đẹp nhưng giao hàng quá chậm và shiper cộc lốc"*).
  - First-token pooling (`[CLS]` / `<s>`) creates an information bottleneck. While self-attention allows early tokens to attend to subsequent tokens, representations of specific aspect keywords (e.g. "giao hàng", "giá", "đóng gói") undergo attenuation across 12 layers.
  - DeBERTa-v3 (`microsoft/mdeberta-v3-base`) does not use Next Sentence Prediction (NSP) during pretraining and utilizes disentangled attention. Empirical NLP literature demonstrates that `[CLS]` pooling is suboptimal for DeBERTa compared to masked mean pooling.
- **Solution 1 — Masked Mean Pooling**:
  - Computes the element-wise average of token representations weighted strictly by the non-padding attention mask:
    $$\text{mask}_{\text{exp}} = \text{attention\_mask}[:, :, \text{None}].\text{float}()$$
    $$\text{pooled} = \frac{\sum_{t=0}^{L-1} (\text{hidden}[:, t, :] \odot \text{mask}_{\text{exp}}[:, t, :])}{\text{clamp}\left(\sum_{t=0}^{L-1} \text{mask}_{\text{exp}}[:, t, :], \min=10^{-9}\right)}$$
  - Properties: $0$ additional parameters, eliminates pad token noise, uniformly incorporates all semantic tokens across the review.
- **Solution 2 — Multi-Head Attention Pooling**:
  - Uses $H$ learnable query projections to dynamically weight token relevance based on sequence contents:
    $$\text{scores}_h = \frac{q_h K_h(\text{hidden})^\top}{\sqrt{d_k}} + \text{mask\_penalty}$$
    $$\text{weights}_h = \text{softmax}(\text{scores}_h, \text{dim}=-1)$$
    $$\text{context}_h = \text{weights}_h V_h(\text{hidden})$$
    $$\text{pooled} = \text{LayerNorm}(\text{Linear}([\text{context}_1 \,\|\, \dots \,\|\, \text{context}_H]))$$
  - Enables the network to learn multiple distinct aggregation subspaces (e.g. general sentiment words vs aspect descriptor tokens).

### 2.2. Head Architecture Bottleneck & Hierarchical Head Connection
- **Observation**: `ml/models/transformer/model.py:40` instantiates 7 isolated linear heads:
  `self.heads = nn.ModuleList([nn.Linear(hidden, t.num_classes) for t in TASK_SPECS])`
  All 7 heads execute concurrently from the same flat `pooled` feature vector:
  `return [head(pooled) for head in self.heads]`
- **Problem**: There is zero inductive bias linking overall document sentiment and aspect-level sentiment.
  - Overall sentiment is easier to learn because all reviews have an overall sentiment label (high sample support across negative, neutral, positive).
  - Rare aspects (`as_price`, `as_service`) have very low frequency in training data. Without contextual guidance from the overall sentiment representation, the model struggles to distinguish subtle polarity shifts in minority aspects.
- **Hierarchical Head Design**:
  - Step 1: Overall Sentiment Head predicts the 3-class overall sentiment from the pooled feature vector and extracts a latent sentiment feature vector $h_{\text{os}} \in \mathbb{R}^{D_{\text{os}}}$.
  - Step 2: The 6 Aspect Sentiment Heads receive the concatenated representation of the pooled feature and the overall sentiment latent representation:
    $$h_{\text{aspect\_in}} = [h_{\text{pooled}} \,\|\, h_{\text{os}}] \in \mathbb{R}^{D + D_{\text{os}}}$$
  - Step 3: Each aspect head projects $h_{\text{aspect\_in}}$ to 4 classes (negative, neutral, positive, absent).
  - Step 4: The output remains `[logits_os, logits_as1, logits_as2, logits_as3, logits_as4, logits_as5, logits_as6]`.

### 2.3. Zero-Breaking Interface Compatibility
- **Observation**:
  - `multitask_loss` expects `logits` to be a `list[Tensor]` of length 7 (`ml/training/losses.py:16-29`).
  - `_predict_loader` iterates over `logits` and applies `softmax` per task (`ml/models/transformer/model.py:86-93`).
  - `calibrate_absent_thresholds` and `decode_probabilities` expect `list[np.ndarray]` where index 0 is $(N, 3)$ and indices 1..6 are $(N, 4)$ (`ml/evaluation/calibration.py:7-43`).
- **Conclusion**: The proposed hierarchical head and pooling modules preserve the exact tensor output signature `list[Tensor]` of length 7. No downstream changes to loss functions, training loops, calibration algorithms, or metric evaluators are needed.

### 2.4. System Parity: Training vs Production Deployment
- **Observation**: `packages/absa_core/absa_core/models/unified_architectures.py:36-40` defines `EncoderMultiTaskNetwork` for inference in the FastAPI application (`apps/api/app/services/absa_service.py`).
- **Conclusion**: When modifying `EncoderMultiTaskNetwork` in `ml/models/transformer/model.py`, the corresponding class in `packages/absa_core/absa_core/models/unified_architectures.py` must be updated synchronously to allow `python -m ml.benchmark --promote-best` and `UnifiedArtifactPredictor` to load the new weights without shape or key mismatches.

---

## 3. Caveats

1. **Parameter & FLOP Overhead**:
   - `MaskedMeanPooling`: Exactly 0 extra parameters, negligible FLOP overhead ($O(L \cdot D)$ element-wise arithmetic).
   - `MultiHeadAttentionPooling`: Adds $4 \times (D \times D)$ linear projections + query parameters (~$3D^2$ params, ~2.3M params for $D=768$).
   - `HierarchicalMultiTaskHead`: Adds $\approx D \times D_{\text{os}} + 6 \times (D + D_{\text{os}}) \times 4$ parameters. For $D=768, D_{\text{os}}=128$, this adds ~100K parameters (<0.1% of base model size).
2. **AMP FP16 & Stability**:
   - When using Multi-Head Attention Pooling or Masked Mean Pooling under FP16 (`GradScaler`), division by sequence mask sum must use `.clamp(min=1e-9)` to prevent `0/0 = NaN` on empty or heavily padded sequences.
   - Attention masking in attention pooling must use $-10000.0$ (or `-1e4`) rather than `-1e9` or `-inf` to avoid FP16 underflow/overflow artifacts in `torch.softmax`.
3. **Configuration Schema**:
   - Model configs (`ml/configs/models/*.yaml`) should expose `pooling_type` (`"masked_mean"`, `"attention"`, `"cls"`) and `head_type` (`"hierarchical"`, `"flat"`) with sane defaults so that older experiments or baseline comparisons remain reproducible.

---

## 4. Conclusion & Concrete Architectural Design

### 4.1. Recommended Modular Pooling Module (`ml/models/transformer/pooling.py`)

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
    """Averages hidden states over valid (non-padding) tokens."""
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, D], attention_mask: [B, L]
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention Pooling over sequence tokens with learnable query vectors."""
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
        # [B, L, H, D_h] -> [B, H, L, D_h]
        k = self.key_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.val_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # [1, H, 1, D_h] -> [B, H, 1, D_h]
        q = self.query.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)

        # Scaled dot-product: [B, H, 1, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            # Mask out padding tokens: attention_mask is [B, L]
            mask_bias = (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -10000.0
            scores = scores + mask_bias

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [B, H, 1, D_h] -> [B, D]
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

---

### 4.2. Recommended Hierarchical Multi-Task Head (`ml/models/transformer/heads.py`)

```python
"""Hierarchical and Flat Task Heads for ABSA."""
from __future__ import annotations

import torch
import torch.nn as nn
from ml.data.schema import TASK_SPECS


class FlatMultiTaskHead(nn.Module):
    """Standard parallel independent classification heads."""
    def __init__(self, hidden_size: int, dropout: float = 0.15):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, t.num_classes) for t in TASK_SPECS])

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h = self.dropout(pooled)
        return [head(h) for head in self.heads]


class HierarchicalMultiTaskHead(nn.Module):
    """Hierarchical head conditioning aspect sentiment classifications on overall sentiment latent features."""
    def __init__(self, hidden_size: int, dropout: float = 0.15, os_latent_dim: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Overall Sentiment branch (3 classes: neg, neu, pos)
        self.os_dense = nn.Sequential(
            nn.Linear(hidden_size, os_latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.os_head = nn.Linear(os_latent_dim, TASK_SPECS[0].num_classes)
        
        # Aspect Sentiment branches (6 aspects, 4 classes each: neg, neu, pos, abs)
        # Conditioned on concatenation of base pooled representation and overall sentiment feature
        combined_dim = hidden_size + os_latent_dim
        self.as_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, t.num_classes)
            ) for t in TASK_SPECS[1:]
        ])

    def forward(self, pooled: torch.Tensor) -> list[torch.Tensor]:
        h_base = self.dropout(pooled)
        h_os = self.os_dense(h_base)
        logits_os = self.os_head(h_os)
        
        # Condition aspect heads on overall sentiment latent
        h_combined = torch.cat([h_base, h_os], dim=-1)
        logits_aspects = [head(h_combined) for head in self.as_heads]
        
        return [logits_os, *logits_aspects]


def build_task_heads(head_type: str, hidden_size: int, dropout: float = 0.15) -> nn.Module:
    h_type = str(head_type).lower().strip()
    if h_type in {"flat", "linear", "independent"}:
        return FlatMultiTaskHead(hidden_size, dropout=dropout)
    elif h_type in {"hierarchical", "cascade", "hier"}:
        return HierarchicalMultiTaskHead(hidden_size, dropout=dropout)
    else:
        raise ValueError(f"Unsupported head_type: '{head_type}'. Available: 'flat', 'hierarchical'")
```

---

### 4.3. Unified `EncoderMultiTaskNetwork` in `ml/models/transformer/model.py`

```python
class EncoderMultiTaskNetwork(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        dropout: float = 0.15,
        pooling_type: str = "masked_mean",
        head_type: str = "hierarchical",
        from_config_only: bool = False
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

---

### 4.4. Configuration Updates
In `ml/configs/models/phobert.yaml`, `mdeberta.yaml`, `xlmr.yaml`:
```yaml
model_name: vinai/phobert-base-v2
pooling_type: masked_mean     # Options: masked_mean, multihead_attention, first_token
head_type: hierarchical       # Options: hierarchical, flat
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

---

## 5. Verification Method

### 5.1. Independent Unit Verification Script
Execute a quick PyTorch tensor verification test locally:
```bash
python -c "
import torch
from ml.models.transformer.pooling import MaskedMeanPooling, MultiHeadAttentionPooling
from ml.models.transformer.heads import HierarchicalMultiTaskHead

B, L, D = 4, 32, 768
hidden = torch.randn(B, L, D)
mask = torch.ones(B, L, dtype=torch.long)
mask[:, 24:] = 0  # simulate padding

mean_p = MaskedMeanPooling()
out_mean = mean_p(hidden, mask)
assert out_mean.shape == (B, D), f'Expected ({B}, {D}), got {out_mean.shape}'

mha_p = MultiHeadAttentionPooling(hidden_size=D, num_heads=4)
out_mha = mha_p(hidden, mask)
assert out_mha.shape == (B, D), f'Expected ({B}, {D}), got {out_mha.shape}'

hier_head = HierarchicalMultiTaskHead(hidden_size=D)
logits = hier_head(out_mean)
assert len(logits) == 7, f'Expected 7 heads, got {len(logits)}'
assert logits[0].shape == (B, 3), f'Expected (B, 3) for OS, got {logits[0].shape}'
for i in range(1, 7):
    assert logits[i].shape == (B, 4), f'Expected (B, 4) for AS {i}, got {logits[i].shape}'
print('All architecture unit tests passed successfully!')
"
```

### 5.2. Test Suite Execution
Run the existing test suite to ensure non-regression across all components:
```bash
pytest tests/unit tests/smoke
```

### 5.3. Remote Execution Compatibility Check
Validate source packaging with the Kaggle CLI tool:
```bash
python -m tools.kaggle_cli prepare-kernel --model phobert --owner test --dataset test/data
```
Verify that `.kaggle_work/kernels/phobert/sentenai_src.zip` packages the new pooling and head modules seamlessly.
