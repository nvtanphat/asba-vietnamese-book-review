"""Deterministic seeding — mirrors seed_everything()/seed_worker()/GLOBAL_GENERATOR
from 05-phobert-balance-experiment-under1mb.ipynb so `python ml/train.py` reproduces
the notebook's sampler/DataLoader behavior instead of just its hyperparameters.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import set_seed

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Strict deterministic algorithms can break some CUDA kernels; keep False for Kaggle T4 stability.
    torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


GLOBAL_GENERATOR = torch.Generator()
GLOBAL_GENERATOR.manual_seed(SEED)
