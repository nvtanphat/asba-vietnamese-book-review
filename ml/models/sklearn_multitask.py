from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .base import ABSABenchmarkModel
from ml.data.schema import TASK_SPECS


class SklearnMultiTaskABSA(ABSABenchmarkModel):
    family = "classical"

    def __init__(self, name: str, estimator: str, config: dict):
        self.name = name
        self.estimator_type = estimator
        self.config = config
        word_cfg = config.get("word_tfidf", {})
        char_cfg = config.get("char_tfidf", {})
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
                min_df=word_cfg.get("min_df", 2),
                max_df=word_cfg.get("max_df", 0.995),
                max_features=word_cfg.get("max_features", 60000),
                sublinear_tf=True,
                strip_accents=None,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=tuple(char_cfg.get("ngram_range", [3, 5])),
                min_df=char_cfg.get("min_df", 2),
                max_features=char_cfg.get("max_features", 60000),
                sublinear_tf=True,
            )),
        ])
        self.models = []
        self.classes_ = []

    def _class_weights(self, y, n_classes):
        mode = self.config.get("class_weight", "balanced_capped")
        if mode in (None, "none", False):
            return None
        counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
        w = np.zeros(n_classes, dtype=float)
        nz = counts > 0
        w[nz] = len(y) / (n_classes * counts[nz])
        if w[nz].size:
            w[nz] /= w[nz].mean()
        w = np.clip(w, 0, float(self.config.get("max_class_weight", 6.0)))
        return {i: float(w[i]) for i in range(n_classes) if counts[i] > 0}

    def _new_estimator(self, class_weight):
        if self.estimator_type == "logistic":
            return LogisticRegression(
                C=float(self.config.get("C", 2.0)),
                max_iter=int(self.config.get("max_iter", 1200)),
                solver="lbfgs",
                class_weight=class_weight,
                random_state=int(self.config.get("seed", 42)),
            )
        return LinearSVC(
            C=float(self.config.get("C", 1.0)),
            class_weight=class_weight,
            random_state=int(self.config.get("seed", 42)),
        )

    def _focal_refit(self, est, x, target, class_weight, n_classes):
        """sklearn's linear solvers have no pluggable loss, so focal loss is approximated by
        iterative sample reweighting: refit a few rounds with sample_weight = (1 - p_true)^gamma,
        which down-weights examples the model already classifies confidently (the same intent
        as focal loss's (1-p_t)^gamma term), combined multiplicatively with class_weight by
        sklearn itself."""
        gamma = float(self.config.get("focal_gamma", 2.0))
        rounds = int(self.config.get("focal_rounds", 3))
        for _ in range(rounds):
            classes = np.asarray(est.classes_, dtype=int)
            proba = self._one_proba(est, classes, x, n_classes)
            p_true = proba[np.arange(len(target)), target]
            sample_weight = (1.0 - np.clip(p_true, 1e-6, 1.0)) ** gamma
            est = self._new_estimator(class_weight)
            est.fit(x, target, sample_weight=sample_weight)
        return est

    def fit(self, train_texts, train_y, val_texts=None, val_y=None, *, output_dir=None, resume=False):
        x = self.vectorizer.fit_transform(list(train_texts))
        self.models, self.classes_ = [], []
        loss_type = self.config.get("loss_type", "ce")
        for i, task in enumerate(TASK_SPECS):
            target = train_y[:, i]
            class_weight = self._class_weights(target, task.num_classes)
            est = self._new_estimator(class_weight)
            est.fit(x, target)
            if loss_type == "focal":
                est = self._focal_refit(est, x, target, class_weight, task.num_classes)
            self.models.append(est)
            self.classes_.append(np.asarray(est.classes_, dtype=int))
        if output_dir:
            self.save(output_dir)
        return self

    def _one_proba(self, model, classes, x, n_classes: int):
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(x)
        else:
            scores = model.decision_function(x)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            raw = softmax(scores, axis=1)
        out = np.zeros((x.shape[0], n_classes), dtype=np.float32)
        out[:, classes] = raw
        out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
        return out

    def predict_proba(self, texts):
        x = self.vectorizer.transform(list(texts))
        return [self._one_proba(m, c, x, t.num_classes) for m, c, t in zip(self.models, self.classes_, TASK_SPECS)]

    def save(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "models": self.models, "classes": self.classes_}, out / "model.joblib")
        (out / "metadata.json").write_text(json.dumps({"name": self.name, "family": self.family, "estimator": self.estimator_type, "config": self.config}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, output_dir: str | Path):
        out = Path(output_dir)
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(meta["name"], meta["estimator"], meta.get("config", {}))
        payload = joblib.load(out / "model.joblib")
        obj.vectorizer, obj.models, obj.classes_ = payload["vectorizer"], payload["models"], payload["classes"]
        return obj
