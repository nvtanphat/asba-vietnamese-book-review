from pathlib import Path
import torch

def save_checkpoint(path, **state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save(state,path)

def load_checkpoint(path,device='cpu'): return torch.load(path,map_location=device)
