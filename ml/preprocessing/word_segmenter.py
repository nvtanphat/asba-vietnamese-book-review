from __future__ import annotations

def segment(text: str, backend: str = "none") -> str:
    if backend == "none": return str(text)
    if backend == "pyvi":
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(str(text))
    raise ValueError(f"Unsupported segmenter: {backend}. Segmenter choice is an ablation, not shared preprocessing.")
