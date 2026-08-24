from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch

from absa_core.preprocessing.pipeline import clean_text_series
from .architectures import ASPECT_COLS
from .unified_architectures import TextCNNNetwork, BiLSTMNetwork, EncoderMultiTaskNetwork


class UnifiedArtifactPredictor:
    """Load the model promoted by ``python -m ml.benchmark --promote-best``.

    Supports every primary benchmark family: sklearn, TextCNN, BiLSTM and shared-head
    pretrained encoders (PhoBERT/XLM-R/mDeBERTa). ViT5 is deliberately not a primary
    promotion candidate because its generative objective is not compute/formulation-equivalent.
    """
    def __init__(self, artifact_dir: str | Path = "artifacts/final", device: str | None = None):
        self.root=Path(artifact_dir);self.model_dir=self.root/"model"
        meta_path=self.root/"metadata.json"
        if not meta_path.exists(): raise FileNotFoundError(f"Unified artifact metadata not found: {meta_path}")
        selected=json.loads(meta_path.read_text(encoding="utf-8"));self.model_name=selected["model"]
        model_meta_path=self.model_dir/"metadata.json"
        self.model_meta=json.loads(model_meta_path.read_text(encoding="utf-8")) if model_meta_path.exists() else {}
        self.family=self.model_meta.get("family",selected.get("leaderboard_row",{}).get("family"))
        th=self.root/"thresholds.json";self.thresholds=json.loads(th.read_text(encoding="utf-8")) if th.exists() else {}
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load()

    def _load(self):
        if self.family=="classical":
            import joblib
            payload=joblib.load(self.model_dir/"model.joblib");self.vectorizer=payload["vectorizer"];self.models=payload["models"];self.classes_=payload["classes"]
        elif self.model_name in {"textcnn","bilstm"}:
            cfg=self.model_meta.get("config",{});self.vocab=self.model_meta["vocab"];self.stoi={t:i for i,t in enumerate(self.vocab)}
            net=TextCNNNetwork(len(self.vocab),cfg) if self.model_name=="textcnn" else BiLSTMNetwork(len(self.vocab),cfg);net.load_state_dict(torch.load(self.model_dir/"model.pt",map_location=self.device));self.model=net.to(self.device).eval();self.max_length=int(cfg.get("max_length",160))
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
        else: raise RuntimeError(f"Unsupported promoted family: {self.family}")

    def _clean(self,texts):
        cleaned=[str(x) for x in clean_text_series(pd.Series(texts),lowercase=True).tolist()]
        segmenter = self.model_meta.get("config", {}).get("word_segmenter", getattr(self, "word_segmenter", "none"))
        if segmenter == "pyvi":
            from pyvi import ViTokenizer
            cleaned=[ViTokenizer.tokenize(x) for x in cleaned]
        return cleaned

    @staticmethod
    def _softmax(x):
        x=np.asarray(x,dtype=float);x=x-x.max(-1,keepdims=True);e=np.exp(x);return e/np.clip(e.sum(-1,keepdims=True),1e-12,None)

    def _sk_probs(self,texts):
        from scipy.special import softmax
        x=self.vectorizer.transform(texts);outs=[];dims=[3,4,4,4,4,4,4]
        for model,classes,n in zip(self.models,self.classes_,dims):
            if hasattr(model,"predict_proba"):raw=model.predict_proba(x)
            else:
                s=model.decision_function(x);s=np.column_stack([-s,s]) if s.ndim==1 else s;raw=softmax(s,axis=1)
            p=np.zeros((x.shape[0],n),dtype=np.float32);p[:,np.asarray(classes,dtype=int)]=raw;p/=np.clip(p.sum(1,keepdims=True),1e-12,None);outs.append(p)
        return outs

    def _encode_word_batch(self,texts):
        ids=[];lengths=[]
        for text in texts:
            arr=[self.stoi.get(tok,1) for tok in str(text).split()][:self.max_length] or [1];lengths.append(len(arr));arr += [0]*(self.max_length-len(arr));ids.append(arr)
        return torch.tensor(ids,dtype=torch.long,device=self.device),torch.tensor(lengths,dtype=torch.long,device=self.device)

    def _probs(self,texts):
        texts=self._clean(texts)
        if self.family=="classical":return self._sk_probs(texts)
        with torch.no_grad():
            if self.model_name in {"textcnn","bilstm"}:
                ids,lengths=self._encode_word_batch(texts);logits=self.model(ids,lengths)
            else:
                enc=self.tokenizer(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt").to(self.device);logits=self.model(enc["input_ids"],enc["attention_mask"])
        return [torch.softmax(x,-1).cpu().numpy() for x in logits]

    def predict(self,texts: list[str] | str) -> list[dict[str,Any]]:
        if isinstance(texts,str):texts=[texts]
        probs=self._probs(texts);results=[]
        for r in range(len(texts)):
            overall_probs=probs[0][r].tolist();aspects={};aspect_probs={}
            for i,col in enumerate(ASPECT_COLS,1):
                p=probs[i][r];presence=float(1.0-p[3]);present=presence>=float(self.thresholds.get(col,0.5));sent=int(np.argmax(p[:3])) if present else -1
                aspects[col]=sent;aspect_probs[col]={"presence":round(presence,3),"sentiment":p[:3].tolist() if present else [0.0,0.0,0.0]}
            results.append({"overall":int(np.argmax(probs[0][r])),"overall_probs":overall_probs,"aspects":aspects,"aspect_probs":aspect_probs})
        return results
