from __future__ import annotations
from ml.models.sklearn_multitask import SklearnMultiTaskABSA


def build(config: dict):
    return SklearnMultiTaskABSA(name="linear_svm", estimator="linear_svm", config=config)
