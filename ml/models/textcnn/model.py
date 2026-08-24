from __future__ import annotations
import torch
import torch.nn as nn
from ml.data.schema import TASK_SPECS
from ml.training.torch_text_trainer import TorchTextABSA


class TextCNNNetwork(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        emb=int(config.get("embedding_dim",200)); channels=int(config.get("channels",128)); kernels=config.get("kernels",[3,4,5]); drop=float(config.get("dropout",0.35))
        self.embedding=nn.Embedding(vocab_size,emb,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(emb,channels,int(k),padding=0) for k in kernels])
        feat=channels*len(kernels)
        self.shared=nn.Sequential(nn.Dropout(drop),nn.Linear(feat,int(config.get("hidden_dim",256))),nn.GELU(),nn.Dropout(drop))
        hidden=int(config.get("hidden_dim",256))
        self.heads=nn.ModuleList([nn.Linear(hidden,t.num_classes) for t in TASK_SPECS])
    def forward(self,input_ids,lengths=None):
        x=self.embedding(input_ids).transpose(1,2)
        feats=[]
        for conv in self.convs:
            k=conv.kernel_size[0]
            if x.size(-1)<k: xk=torch.nn.functional.pad(x,(0,k-x.size(-1)))
            else: xk=x
            feats.append(torch.relu(conv(xk)).amax(dim=-1))
        h=self.shared(torch.cat(feats,dim=1))
        return [head(h) for head in self.heads]


def build(config):
    return TorchTextABSA("textcnn", lambda vocab,cfg: TextCNNNetwork(vocab,cfg), config)
