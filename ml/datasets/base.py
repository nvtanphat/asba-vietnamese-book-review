from dataclasses import dataclass
import numpy as np
@dataclass
class BenchmarkBatch:
    texts: list[str]
    labels: np.ndarray
