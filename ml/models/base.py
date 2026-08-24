from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class ABSABenchmarkModel(ABC):
    name: str
    family: str

    @abstractmethod
    def fit(self, train_texts, train_y: np.ndarray, val_texts=None, val_y=None, *, output_dir: str | Path | None = None, resume: bool = False): ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> list[np.ndarray]:
        """Return seven probability arrays: overall (N,3), six aspect arrays (N,4)."""

    @abstractmethod
    def save(self, output_dir: str | Path): ...

    def parameter_count(self) -> int | None:
        return None
