from __future__ import annotations
import importlib

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
