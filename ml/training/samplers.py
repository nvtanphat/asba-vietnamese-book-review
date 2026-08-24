from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler

def joint_label_sampler(labels, seed=42, temperature=0.5):
    sig=[tuple(map(int,row)) for row in labels];counts=Counter(sig);w=torch.tensor([counts[s]**(-temperature) for s in sig],dtype=torch.double);return WeightedRandomSampler(w,len(w),replacement=True,generator=torch.Generator().manual_seed(seed))
