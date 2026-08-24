from __future__ import annotations
from ml.models.sklearn_multitask import SklearnMultiTaskABSA


def build(config: dict):
    return SklearnMultiTaskABSA(name="logistic", estimator="logistic", config=config)
