from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.data.schema import TASK_SPECS
from ml.evaluation.metrics import evaluate_predictions
from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
from ml.models.base import ABSABenchmarkModel
from ml.training.losses import multitask_loss
from ml.utils.seed import seed_everything


class Vocabulary:
    def __init__(self, min_freq=2, max_size=50000):
        self.min_freq, self.max_size = min_freq, max_size
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {t:i for i,t in enumerate(self.itos)}

    @staticmethod
    def tokenize(text):
        return str(text).split()

    def fit(self, texts):
        counts = Counter(tok for text in texts for tok in self.tokenize(text))
        for tok, count in counts.most_common(self.max_size - len(self.itos)):
            if count < self.min_freq:
                break
            self.stoi[tok] = len(self.itos); self.itos.append(tok)
        return self

    def encode(self, text, max_length):
        ids = [self.stoi.get(t, 1) for t in self.tokenize(text)][:max_length]
        if not ids: ids = [1]
        length = len(ids)
        ids += [0] * (max_length - length)
        return ids, length


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary, max_length: int):
        self.texts, self.labels, self.vocab, self.max_length = list(texts), labels, vocab, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ids, length = self.vocab.encode(self.texts[idx], self.max_length)
        item = {"input_ids": torch.tensor(ids, dtype=torch.long), "length": torch.tensor(length, dtype=torch.long)}
        if self.labels is not None: item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def class_weights(y: np.ndarray, n_classes: int, max_weight: float = 6.0):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = np.zeros(n_classes, dtype=np.float32)
    nz = counts > 0
    weights[nz] = len(y) / (n_classes * counts[nz])
    if weights[nz].size:
        weights[nz] /= weights[nz].mean()
    return torch.tensor(np.clip(weights, 0, max_weight), dtype=torch.float32)


class TorchTextABSA(ABSABenchmarkModel):
    family = "deep_learning"
    def __init__(self, name, network_factory, config):
        self.name, self.network_factory, self.config = name, network_factory, config
        self.vocab = Vocabulary(config.get("min_freq", 2), config.get("vocab_size", 50000))
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = []

    def _normalize_model_input(self, texts):
        texts = [str(t) for t in texts]
        if self.config.get("word_segmenter", "none") == "pyvi":
            try:
                from pyvi import ViTokenizer
            except ImportError as exc:
                raise RuntimeError("word_segmenter=pyvi requires pyvi") from exc
            texts = [ViTokenizer.tokenize(t) for t in texts]
        return texts

    def _loader(self, texts, y, shuffle=False):
        ds = TextDataset(self._normalize_model_input(texts), y, self.vocab, int(self.config.get("max_length", 160)))
        gen = torch.Generator().manual_seed(int(self.config.get("seed", 42)))
        return DataLoader(ds, batch_size=int(self.config.get("batch_size", 64)), shuffle=shuffle, num_workers=0, generator=gen)

    def _predict_loader(self, loader):
        probs = [[] for _ in TASK_SPECS]
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch["input_ids"].to(self.device), batch["length"].to(self.device))
                for i, x in enumerate(logits): probs[i].append(torch.softmax(x, -1).cpu().numpy())
        return [np.concatenate(p, axis=0) if p else np.empty((0,t.num_classes)) for p,t in zip(probs,TASK_SPECS)]

    def fit(self, train_texts, train_y, val_texts=None, val_y=None, *, output_dir=None, resume=False):
        seed = int(self.config.get("seed", 42)); seed_everything(seed)
        normalized_train = self._normalize_model_input(train_texts)
        self.vocab.fit(normalized_train)
        self.model = self.network_factory(len(self.vocab.itos), self.config).to(self.device)
        weights = [class_weights(train_y[:, i], t.num_classes).to(self.device) for i,t in enumerate(TASK_SPECS)]
        task_weights = self.config.get("task_weights", None)
        accum_steps = max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.config.get("lr", 1e-3)), weight_decay=float(self.config.get("weight_decay", 1e-4)))
        train_loader = self._loader(train_texts, train_y, shuffle=True)
        val_loader = self._loader(val_texts, val_y, shuffle=False) if val_texts is not None else None
        epochs = int(self.config.get("epochs", 20)); patience = int(self.config.get("patience", 4))
        start_epoch, best, bad = 0, -1.0, 0
        out = Path(output_dir) if output_dir else None
        if out: out.mkdir(parents=True, exist_ok=True)
        last_path = out / "last.pt" if out else None
        if resume and last_path and last_path.exists():
            ckpt = torch.load(last_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch, best, bad = ckpt["epoch"] + 1, ckpt.get("best", -1), ckpt.get("bad", 0)

        for epoch in range(start_epoch, epochs):
            self.model.train(); losses=[]
            optimizer.zero_grad(set_to_none=True)
            for step_idx, batch in enumerate(train_loader):
                logits = self.model(batch["input_ids"].to(self.device), batch["length"].to(self.device))
                labels = batch["labels"].to(self.device)
                loss = multitask_loss(logits, labels, weights=weights, task_weights=task_weights, loss_type=self.config.get("loss_type", "ce"), gamma=float(self.config.get("focal_gamma", 2.0)), label_smoothing=float(self.config.get("label_smoothing", 0.0)))
                loss_step = loss / accum_steps
                loss_step.backward()
                losses.append(float(loss.detach().cpu()))
                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            row={"epoch":epoch+1,"train_loss":float(np.mean(losses))}

            if val_loader is not None:
                vp=self._predict_loader(val_loader); thresholds=calibrate_absent_thresholds(vp, np.asarray(val_y)); pred=decode_probabilities(vp, thresholds); metrics=evaluate_predictions(np.asarray(val_y), pred)
                row.update({"val_"+k:v for k,v in metrics.items()}); score=metrics["f1_combined"]
                if score > best:
                    best, bad = score, 0
                    if out: torch.save(self.model.state_dict(), out / "best.pt")
                else:
                    bad += 1
            self.history.append(row)
            if last_path:
                torch.save({"model":self.model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"best":best,"bad":bad}, last_path)
            if bad >= patience: break
        if out and (out / "best.pt").exists(): self.model.load_state_dict(torch.load(out / "best.pt", map_location=self.device))
        if out: self.save(out)
        return self

    def predict_proba(self, texts):
        if self.model is None: raise RuntimeError("Model is not fitted")
        return self._predict_loader(self._loader(texts, None, shuffle=False))

    def save(self, output_dir):
        out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model.pt")
        meta={"name":self.name,"family":self.family,"config":self.config,"vocab":self.vocab.itos,"history":self.history}
        (out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")

    def parameter_count(self):
        return int(sum(p.numel() for p in self.model.parameters())) if self.model is not None else None
