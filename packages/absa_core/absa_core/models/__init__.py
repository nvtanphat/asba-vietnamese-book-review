"""Lazy model exports so preprocessing/data tooling does not eagerly import Transformers."""

__all__ = ["ABSAPredictor", "UnifiedArtifactPredictor", "EnsembleOnnxPredictor"]

def __getattr__(name):
    if name == "ABSAPredictor":
        from .predictor import ABSAPredictor
        return ABSAPredictor
    if name == "UnifiedArtifactPredictor":
        from .unified_predictor import UnifiedArtifactPredictor
        return UnifiedArtifactPredictor
    if name == "EnsembleOnnxPredictor":
        from .ensemble_onnx_predictor import EnsembleOnnxPredictor
        return EnsembleOnnxPredictor
    raise AttributeError(name)
