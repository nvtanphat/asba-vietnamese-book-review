"""Lazy model exports so preprocessing/data tooling does not eagerly import Transformers."""

__all__ = ["ABSAPredictor", "UnifiedArtifactPredictor"]

def __getattr__(name):
    if name == "ABSAPredictor":
        from .predictor import ABSAPredictor
        return ABSAPredictor
    if name == "UnifiedArtifactPredictor":
        from .unified_predictor import UnifiedArtifactPredictor
        return UnifiedArtifactPredictor
    raise AttributeError(name)
