# Challenger 2 Review Report: M1 Integration & Parity Stress Testing

**Verdict**: **REQUEST_CHANGES**

---

## 1. Observation

### 1.1. Empirical Stress Testing Executed
A comprehensive empirical test suite was developed and executed in `tests/unit/test_m1_parity_stress.py`:
- **Forward Pass Matrix**: Tested all 6 combinations of `pooling_type` (`first_token`, `masked_mean`, `multihead_attention`) $\times$ `head_type` (`flat`, `hierarchical`) across batch sizes $\{1, 2, 7, 16\}$ and sequence lengths $\{8, 32, 64, 128\}$. All shapes matched the exact 7-task specification: index 0 is $(B, 3)$, indices 1..6 are $(B, 4)$.
- **Boundary Conditions**: Tested sequence length 1, 100% padded sequences (all zeros mask), single-token active masks, and corrupted pad token values. Epsilon-clamping ($\epsilon = 10^{-4}$) and $-10000.0$ mask bias successfully prevented division-by-zero, NaN, and Inf.
- **Loss Backward & Gradient Flow**: Tested backpropagation with CrossEntropy and Focal Loss across all active layers (embeddings, transformer encoder, pooler projections/query/LayerNorm, hierarchical latent dense + aspect heads). Gradients were non-zero, finite, and free of NaNs.
- **State Dict Parity**: Verified 100% exact key and shape equality between `ml.models.transformer.model.EncoderMultiTaskNetwork` and `packages/absa_core/absa_core/models/unified_architectures.EncoderMultiTaskNetwork` with `strict=True` across all 6 pooling-head combinations.

### 1.2. Defect Discovered: Predictor Config Parameter Propagation Bug
In `packages/absa_core/absa_core/models/unified_predictor.py` line 43:
```python
        elif self.family=="pretrained_encoder":
            from transformers import AutoTokenizer
            cfg=self.model_meta.get("config",{})
            self.tokenizer=AutoTokenizer.from_pretrained(self.model_dir/"tokenizer")
            net=EncoderMultiTaskNetwork(str(self.model_dir/"encoder"),float(cfg.get("dropout",0.15)))
            net.load_state_dict(torch.load(self.model_dir/"model.pt",map_location=self.device))
            self.model=net.to(self.device).eval();self.max_length=int(cfg.get("max_length",160));self.word_segmenter=cfg.get("word_segmenter","none")
```
- `EncoderMultiTaskNetwork` in `absa_core.models.unified_architectures` has the signature:
  `EncoderMultiTaskNetwork(config_dir: str, dropout: float = 0.15, pooling_type: str = "masked_mean", head_type: str = "hierarchical")`
- `UnifiedArtifactPredictor._load()` instantiates `EncoderMultiTaskNetwork(str(self.model_dir/"encoder"), float(cfg.get("dropout", 0.15)))` without passing `pooling_type` or `head_type` from `cfg` (`self.model_meta.get("config", {})`).

---

## 2. Logic Chain

1. **State Dict Compatibility Failure on Promoted Models**:
   - If a model is trained with `pooling_type: "multihead_attention"`, its saved state dict contains learnable attention weights: `pooler.query`, `pooler.key_proj.weight`, `pooler.val_proj.weight`, `pooler.out_proj.weight`, `pooler.layer_norm.weight`, etc.
   - Because `UnifiedArtifactPredictor` fails to pass `pooling_type=cfg.get("pooling_type", "masked_mean")`, `EncoderMultiTaskNetwork` defaults to `MaskedMeanPooling` (0 parameters).
   - During `net.load_state_dict(...)`, PyTorch raises `RuntimeError: Unexpected key(s) in state_dict: "pooler.query", "pooler.key_proj.weight", ...`.

2. **Head Type Deserialization Failure**:
   - If a model is trained with `head_type: "flat"`, its saved state dict contains `task_head.heads.0..6`.
   - Because `UnifiedArtifactPredictor` defaults to `HierarchicalMultiTaskHead`, loading the state dict raises `RuntimeError: Missing key(s) in state_dict: "task_head.os_dense.0.weight" ... Unexpected key(s) in state_dict: "task_head.heads.0.weight" ...`.

3. **Silent Production Divergence on First-Token Pooling**:
   - If a model is trained with `pooling_type: "first_token"`, both `FirstTokenPooling` and `MaskedMeanPooling` have no trainable parameters.
   - `load_state_dict` does not raise an exception, but `UnifiedArtifactPredictor` silently executes `MaskedMeanPooling` during production serving instead of `FirstTokenPooling`, causing silent feature distortion and divergence from validation metrics.

4. **Empirical Reproduction**:
   - Replicated via `test_unified_predictor_config_propagation_check` in `tests/unit/test_m1_parity_stress.py`.

---

## 3. Caveats

- For models trained with the default configuration (`pooling_type: masked_mean` and `head_type: hierarchical`), `UnifiedArtifactPredictor` loads and executes successfully because the hardcoded defaults in `EncoderMultiTaskNetwork` match.
- However, failing to propagate configuration violates the modular design contract and prevents non-default architecture exploration (such as `multihead_attention` experiments).

---

## 4. Conclusion & Required Action

**Verdict**: **REQUEST_CHANGES**

### Required Code Change:
In `packages/absa_core/absa_core/models/unified_predictor.py` (lines 41–44), pass `pooling_type` and `head_type` from `cfg`:
```python
        elif self.family=="pretrained_encoder":
            from transformers import AutoTokenizer
            cfg=self.model_meta.get("config",{})
            self.tokenizer=AutoTokenizer.from_pretrained(self.model_dir/"tokenizer")
            net=EncoderMultiTaskNetwork(
                str(self.model_dir/"encoder"),
                float(cfg.get("dropout",0.15)),
                pooling_type=str(cfg.get("pooling_type", "masked_mean")),
                head_type=str(cfg.get("head_type", "hierarchical")),
            )
            net.load_state_dict(torch.load(self.model_dir/"model.pt",map_location=self.device))
            self.model=net.to(self.device).eval();self.max_length=int(cfg.get("max_length",160));self.word_segmenter=cfg.get("word_segmenter","none")
```

---

## 5. Verification Method

Run the full pytest suite:
```bash
python -m pytest tests/ -v
```

Specifically run the M1 parity and stress suite:
```bash
python -m pytest tests/unit/test_m1_parity_stress.py -v
```
