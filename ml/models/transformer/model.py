from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from ml.data.schema import TASK_SPECS
from ml.evaluation.metrics import evaluate_predictions
from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
from ml.models.base import ABSABenchmarkModel
from ml.models.transformer.heads import build_task_heads
from ml.models.transformer.pooling import build_pooling_layer
from ml.training.losses import class_balanced_weights, two_stage_multitask_loss, two_stage_multitask_weights
from ml.training.torch_text_trainer import class_weights
from ml.utils.seed import seed_everything

MODEL_NAMES = {
    "phobert": "vinai/phobert-base-v2",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
}


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
        self.pooler = build_pooling_layer(pooling_type, hidden, dropout=dropout)
        self.task_head = build_task_heads(head_type, hidden, dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self.pooler(hidden, attention_mask)
        return self.task_head(pooled)


class _TextDataset(Dataset):
    def __init__(self, texts, y=None): self.texts=list(texts); self.y=y
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], None if self.y is None else self.y[i]


class TransformerMultiTaskABSA(ABSABenchmarkModel):
    family = "pretrained_encoder"
    def __init__(self, name: str, config: dict):
        self.name, self.config = name, config
        self.model_name = config.get("model_name", MODEL_NAMES[name])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=config.get("use_fast", True))
        dropout = float(config.get("dropout", 0.15))
        pooling_type = str(config.get("pooling_type", "masked_mean"))
        head_type = str(config.get("head_type", "hierarchical"))
        self.model = EncoderMultiTaskNetwork(
            self.model_name,
            dropout=dropout,
            pooling_type=pooling_type,
            head_type=head_type,
        ).to(self.device)
        self.history = []

    def _normalize_model_input(self, texts):
        texts = [str(t) for t in texts]
        if self.name == "phobert" and self.config.get("word_segmenter") == "pyvi":
            try:
                from pyvi import ViTokenizer
                texts = [ViTokenizer.tokenize(t) for t in texts]
            except Exception as exc:
                raise RuntimeError("word_segmenter=pyvi requires pyvi") from exc
        prefix = self.config.get("aspect_prompt_prefix")
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        return texts

    def _collate(self, batch):
        texts=[x[0] for x in batch]; labels=[x[1] for x in batch]
        enc=self.tokenizer(self._normalize_model_input(texts),padding=True,truncation=True,max_length=int(self.config.get("max_length",160)),return_tensors="pt")
        if labels[0] is not None: enc["labels"]=torch.tensor(np.asarray(labels),dtype=torch.long)
        return enc

    def _joint_balanced_weights(self, train_y):
        """Per-row sample weights that oversample rare (overall_sentiment, #aspects_present)
        combinations, matching the "joint_balanced" sampler from the specialized legacy
        PhoBERT trainer's winning config. Complements the loss-level class weighting with a
        data-level intervention: rare joint groups get seen more often per epoch, not just
        scored more heavily when they are."""
        temperature = float(self.config.get("sampler_temperature", 0.5))
        cap = float(self.config.get("sampler_weight_cap", 4.0))
        present_count = (train_y[:, 1:] != 3).sum(axis=1)
        groups = [f"{s}|{c}" for s, c in zip(train_y[:, 0].tolist(), present_count.tolist())]
        counts = Counter(groups)
        weights = np.array([counts[g] ** (-temperature) for g in groups], dtype=float)
        weights = weights / weights.mean()
        weights = np.clip(weights, a_min=None, a_max=cap)
        weights = weights / weights.mean()
        return weights

    def _loader(self,texts,y=None,shuffle=False):
        gen=torch.Generator().manual_seed(int(self.config.get("seed",42)))
        sampler=None
        if shuffle and y is not None and bool(self.config.get("joint_balanced_sampler", False)):
            w=self._joint_balanced_weights(y)
            sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),num_samples=len(w),replacement=True,generator=gen)
        return DataLoader(_TextDataset(texts,y),batch_size=int(self.config.get("batch_size",16)),shuffle=(shuffle and sampler is None),sampler=sampler,num_workers=0,collate_fn=self._collate,generator=gen)

    def _predict_loader(self,loader):
        probs=[[] for _ in TASK_SPECS]; self.model.eval()
        with torch.no_grad():
            for batch in loader:
                logits=self.model(batch["input_ids"].to(self.device),batch["attention_mask"].to(self.device))
                for i,x in enumerate(logits): probs[i].append(torch.softmax(x,-1).cpu().numpy())
        return [np.concatenate(p,0) if p else np.empty((0,t.num_classes)) for p,t in zip(probs,TASK_SPECS)]

    def _loss(self, logits, labels, weights, task_weights, aspect_weights):
        return two_stage_multitask_loss(
            logits, labels, self.model.task_head, weights, aspect_weights, task_weights=task_weights,
            loss_type=self.config.get("loss_type", "ce"), gamma=float(self.config.get("focal_gamma", 2.0)), label_smoothing=float(self.config.get("label_smoothing", 0.05)),
            sentiment_loss_weight=float(self.config.get("sentiment_loss_weight", 0.5)), aspect_loss_weight=float(self.config.get("aspect_loss_weight", 0.5)),
            stage1_weight=float(self.config.get("stage1_weight", 0.25)), stage2_weight=float(self.config.get("stage2_weight", 0.75)),
            focal_gamma_present=float(self.config.get("focal_gamma_present", 2.5)), aspect_label_smoothing=float(self.config.get("aspect_label_smoothing", 0.1)),
        )

    def fit(self,train_texts,train_y,val_texts=None,val_y=None,*,output_dir=None,resume=False):
        seed=int(self.config.get("seed",42)); seed_everything(seed)
        out=Path(output_dir) if output_dir else None
        if out: out.mkdir(parents=True,exist_ok=True)
        train_loader=self._loader(train_texts,train_y,True); val_loader=self._loader(val_texts,val_y,False) if val_texts is not None else None
        weights=[class_weights(train_y[:,i],t.num_classes,float(self.config.get("max_class_weight",6.0))).to(self.device) for i,t in enumerate(TASK_SPECS)]
        task_weights=self.config.get("task_weights", None)
        is_two_stage=self.model.head_type in {"two_stage","presence_sentiment","presence_polarity"}
        if is_two_stage and bool(self.config.get("sentiment_class_balanced_weight", False)):
            # Optional: match the legacy trainer's class-balanced (effective-number) weighting
            # on the overall-sentiment task too, not just the aspect presence/sentiment stages.
            # Off by default: bundled with other legacy-matching hyperparameters it regressed
            # phobert vs. the proven two_stage+joint_sampler config, and was never isolated.
            weights[0]=class_balanced_weights(train_y[:,0],TASK_SPECS[0].num_classes,beta=float(self.config.get("class_balanced_beta",0.999))).to(self.device)
        aspect_weights=two_stage_multitask_weights(train_y, beta=float(self.config.get("class_balanced_beta",0.999)), absent_scale=float(self.config.get("absent_weight_scale",0.2))) if is_two_stage else None
        accum_steps=max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=float(self.config.get("lr",2e-5)),weight_decay=float(self.config.get("weight_decay",0.01)))
        epochs=int(self.config.get("epochs",6))
        total_opt_steps=max(1, (len(train_loader) + accum_steps - 1) // accum_steps * epochs)
        warm=int(total_opt_steps*float(self.config.get("warmup_ratio",0.1)))
        if str(self.config.get("lr_scheduler_type","linear")).lower()=="cosine":
            scheduler=get_cosine_schedule_with_warmup(optimizer,warm,total_opt_steps)
        else:
            scheduler=get_linear_schedule_with_warmup(optimizer,warm,total_opt_steps)
        scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and bool(self.config.get("fp16",True)))
        start,best,bad=0,-1.0,0; patience=int(self.config.get("patience",2)); last=out/"last.pt" if out else None
        if resume and last and last.exists():
            ck=torch.load(last,map_location=self.device); self.model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"]); start=ck["epoch"]+1;best=ck.get("best",-1);bad=ck.get("bad",0)
        for epoch in range(start,epochs):
            self.model.train(); losses=[]
            optimizer.zero_grad(set_to_none=True)
            for step_idx, batch in enumerate(train_loader):
                labels=batch.pop("labels").to(self.device); ids=batch["input_ids"].to(self.device); mask=batch["attention_mask"].to(self.device)
                with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=scaler.is_enabled()):
                    logits=self.model(ids,mask)
                    loss=self._loss(logits,labels,weights,task_weights,aspect_weights)
                    loss_step=loss/accum_steps
                scaler.scale(loss_step).backward()
                losses.append(float(loss.detach().cpu()))
                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),float(self.config.get("max_grad_norm",1.0)))
                    scaler.step(optimizer); scaler.update(); scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            row={"epoch":epoch+1,"train_loss":float(np.mean(losses))}

            if val_loader is not None:
                vp=self._predict_loader(val_loader); thresholds=calibrate_absent_thresholds(vp, np.asarray(val_y)); pred=decode_probabilities(vp, thresholds); met=evaluate_predictions(np.asarray(val_y),pred); row.update({"val_"+k:v for k,v in met.items()}); score=met["f1_combined"]
                if score>best:
                    best,bad=score,0
                    if out: torch.save(self.model.state_dict(),out/"best.pt")
                else: bad+=1
            self.history.append(row)
            if last: torch.save({"model":self.model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"best":best,"bad":bad},last)
            if bad>=patience: break
        if out and (out/"best.pt").exists(): self.model.load_state_dict(torch.load(out/"best.pt",map_location=self.device))
        if out:self.save(out)
        return self

    def predict_proba(self,texts): return self._predict_loader(self._loader(texts,None,False))

    def save(self,output_dir):
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        # Save config/tokenizer locally; full encoder+heads weights live in model.pt.
        (out/"encoder").mkdir(exist_ok=True); self.model.encoder.config.save_pretrained(out/"encoder"); self.tokenizer.save_pretrained(out/"tokenizer")
        torch.save(self.model.state_dict(),out/"model.pt")
        (out/"metadata.json").write_text(json.dumps({"name":self.name,"family":self.family,"model_name":self.model_name,"config":self.config,"history":self.history},ensure_ascii=False,indent=2),encoding="utf-8")

    def parameter_count(self): return int(sum(p.numel() for p in self.model.parameters()))


def build(config):
    return TransformerMultiTaskABSA(config["name"],config)
