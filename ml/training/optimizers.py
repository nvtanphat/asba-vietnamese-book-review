import torch
def adamw(parameters,lr,weight_decay=0.01): return torch.optim.AdamW(parameters,lr=lr,weight_decay=weight_decay)
