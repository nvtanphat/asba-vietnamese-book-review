from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from ml.data.schema import ASPECT_COLS, SENTIMENT_NAMES, ASPECT_SENTIMENT_NAMES, TASK_SPECS
from ml.evaluation.metrics import evaluate_predictions
from ml.models.base import ABSABenchmarkModel
from ml.training.losses import sequence_focal_loss
from ml.utils.seed import seed_everything

DEFAULT_MODEL = "VietAI/vit5-base"
_INV_SENT = {v:k for k,v in SENTIMENT_NAMES.items()}
_INV_ASP = {v:k for k,v in ASPECT_SENTIMENT_NAMES.items()}


def format_target(labels) -> str:
    # ":" not "=": VietAI/vit5-base's sentencepiece tokenizer silently drops "=" on
    # decode (round-trip tested), which made every generated target unparseable.
    parts=[f"sentiment:{SENTIMENT_NAMES[int(labels[0])]}"]
    for i,col in enumerate(ASPECT_COLS,1): parts.append(f"{col}:{ASPECT_SENTIMENT_NAMES[int(labels[i])]}")
    return "; ".join(parts)


def parse_target(text: str) -> np.ndarray:
    pairs=dict(re.findall(r"([a-z_]+)\s*:\s*(negative|neutral|positive|absent)",str(text).lower()))
    out=np.full(7,3,dtype=int); out[0]=_INV_SENT.get(pairs.get("sentiment","neutral"),1)
    for i,col in enumerate(ASPECT_COLS,1): out[i]=_INV_ASP.get(pairs.get(col,"absent"),3)
    return out


class _DS(Dataset):
    def __init__(self,texts,y=None): self.texts=list(texts); self.y=y
    def __len__(self): return len(self.texts)
    def __getitem__(self,i): return self.texts[i], None if self.y is None else self.y[i]


class ViT5GenerativeABSA(ABSABenchmarkModel):
    family="generative"
    def __init__(self,config):
        self.name="vit5";self.config=config;self.model_name=config.get("model_name",DEFAULT_MODEL);self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # use_fast=False: the fast T5 tokenizer's vocab-conversion path crashes with
        # KeyError: 0 on some transformers builds (Kaggle's included) for this checkpoint;
        # the slow/legacy sentencepiece tokenizer is unaffected and produces identical tokens.
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if bool(config.get("lora",True)):
            try:
                from peft import LoraConfig, TaskType, get_peft_model
                lc=LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=int(config.get("lora_r",8)),lora_alpha=int(config.get("lora_alpha",16)),lora_dropout=float(config.get("lora_dropout",0.05)),target_modules=config.get("target_modules",["q","v"]))
                self.model=get_peft_model(self.model,lc)
            except ImportError as exc:
                raise RuntimeError("ViT5 LoRA requires `peft`; install the training dependencies.") from exc
        self.model.to(self.device);self.history=[]

    def _collate(self,batch):
        texts=[f"absa: {x[0]}" for x in batch]; ys=[x[1] for x in batch]
        enc=self.tokenizer(texts,padding=True,truncation=True,max_length=int(self.config.get("max_length",192)),return_tensors="pt")
        if ys[0] is not None:
            targets=[format_target(y) for y in ys]
            lab=self.tokenizer(text_target=targets,padding=True,truncation=True,max_length=int(self.config.get("target_max_length",96)),return_tensors="pt")["input_ids"]
            lab[lab==self.tokenizer.pad_token_id]=-100;enc["labels"]=lab
        return enc

    def _loader(self,texts,y=None,shuffle=False,weighted=False):
        ds=_DS(texts,y); sampler=None
        if weighted and y is not None:
            sig=[tuple(row.tolist()) for row in np.asarray(y)]; from collections import Counter
            counts=Counter(sig); w=torch.tensor([counts[s]**-0.5 for s in sig],dtype=torch.double);sampler=WeightedRandomSampler(w,len(w),replacement=True,generator=torch.Generator().manual_seed(int(self.config.get("seed",42))))
        return DataLoader(ds,batch_size=int(self.config.get("batch_size",4)),shuffle=shuffle and sampler is None,sampler=sampler,num_workers=0,collate_fn=self._collate)

    def _generate(self,texts):
        loader=self._loader(texts,None); rows=[];self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch={k:v.to(self.device) for k,v in batch.items()}
                ids=self.model.generate(**batch,max_new_tokens=int(self.config.get("max_new_tokens",96)),num_beams=int(self.config.get("num_beams",2)))
                rows.extend(parse_target(t) for t in self.tokenizer.batch_decode(ids,skip_special_tokens=True))
        return np.vstack(rows) if rows else np.empty((0,7),dtype=int)

    def fit(self,train_texts,train_y,val_texts=None,val_y=None,*,output_dir=None,resume=False):
        seed_everything(int(self.config.get("seed",42)));out=Path(output_dir) if output_dir else None
        if out: out.mkdir(parents=True,exist_ok=True)
        loader=self._loader(train_texts,train_y,weighted=bool(self.config.get("balanced_sampler",True)))
        accum_steps=max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        params=[p for p in self.model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=float(self.config.get("lr",2e-4)),weight_decay=float(self.config.get("weight_decay",0.01)))
        epochs=int(self.config.get("epochs",5))
        total_opt_steps=max(1, (len(loader) + accum_steps - 1) // accum_steps * epochs)
        sched=get_linear_schedule_with_warmup(opt,int(total_opt_steps*float(self.config.get("warmup_ratio",0.05))),total_opt_steps)
        start,best,bad=0,-1.0,0;patience=int(self.config.get("patience",2));last=out/"last.pt" if out else None
        loss_type=self.config.get("loss_type","ce");gamma=float(self.config.get("focal_gamma",2.0))
        if resume and last and last.exists():
            ck=torch.load(last,map_location=self.device);self.model.load_state_dict(ck["model"],strict=False);opt.load_state_dict(ck["optimizer"]);sched.load_state_dict(ck["scheduler"]);start=ck["epoch"]+1;best=ck.get("best",-1);bad=ck.get("bad",0)
        for epoch in range(start,epochs):
            self.model.train();losses=[]
            opt.zero_grad(set_to_none=True)
            for step_idx, batch in enumerate(loader):
                batch={k:v.to(self.device) for k,v in batch.items()}
                res=self.model(**batch)
                loss=sequence_focal_loss(res.logits,batch["labels"],gamma=gamma) if loss_type=="focal" else res.loss
                loss_step=loss/accum_steps
                loss_step.backward()
                losses.append(float(loss.detach().cpu()))
                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();sched.step()
                    opt.zero_grad(set_to_none=True)
            row={"epoch":epoch+1,"train_loss":float(np.mean(losses))}

            if val_texts is not None:
                pred=self._generate(val_texts);met=evaluate_predictions(np.asarray(val_y),pred);row.update({"val_"+k:v for k,v in met.items()});score=met["f1_combined"]
                if score>best:
                    best,bad=score,0
                    if out:torch.save(self.model.state_dict(),out/"best.pt")
                else:bad+=1
            self.history.append(row)
            if last:torch.save({"model":self.model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"epoch":epoch,"best":best,"bad":bad},last)
            if bad>=patience:break
        if out and (out/"best.pt").exists():
            self.model.load_state_dict(torch.load(out/"best.pt",map_location=self.device),strict=False)
        if out:self.save(out)
        return self

    def predict_proba(self,texts):
        pred=self._generate(texts);probs=[]
        for j,t in enumerate(TASK_SPECS):
            p=np.full((len(pred),t.num_classes),1e-4,dtype=np.float32);p[np.arange(len(pred)),pred[:,j]]=1.0;p/=p.sum(1,keepdims=True);probs.append(p)
        return probs

    def save(self,output_dir):
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);self.model.save_pretrained(out/"model");self.tokenizer.save_pretrained(out/"tokenizer");(out/"metadata.json").write_text(json.dumps({"name":self.name,"family":self.family,"model_name":self.model_name,"config":self.config,"history":self.history,"primary_benchmark_eligible":False},ensure_ascii=False,indent=2),encoding="utf-8")
    def parameter_count(self):return int(sum(p.numel() for p in self.model.parameters()))


def build(config): return ViT5GenerativeABSA(config)
