from __future__ import annotations
import torch
import torch.nn as nn
from ml.data.schema import TASK_SPECS
from ml.training.torch_text_trainer import TorchTextABSA


class BiLSTMNetwork(nn.Module):
    def __init__(self,vocab_size,config):
        super().__init__()
        emb=int(config.get("embedding_dim",200)); hidden=int(config.get("hidden_dim",192)); layers=int(config.get("num_layers",2)); drop=float(config.get("dropout",0.35))
        self.embedding=nn.Embedding(vocab_size,emb,padding_idx=0)
        self.lstm=nn.LSTM(emb,hidden,num_layers=layers,batch_first=True,bidirectional=True,dropout=drop if layers>1 else 0)
        self.proj=nn.Sequential(nn.LayerNorm(hidden*4),nn.Linear(hidden*4,hidden*2),nn.GELU(),nn.Dropout(drop))
        self.heads=nn.ModuleList([nn.Linear(hidden*2,t.num_classes) for t in TASK_SPECS])
    def forward(self,input_ids,lengths):
        x=self.embedding(input_ids)
        packed=nn.utils.rnn.pack_padded_sequence(x,lengths.clamp(min=1).cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.lstm(packed); out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=input_ids.size(1))
        mask=torch.arange(input_ids.size(1),device=input_ids.device)[None,:] < lengths[:,None]
        mean=(out*mask.unsqueeze(-1)).sum(1)/lengths.clamp(min=1).unsqueeze(-1)
        maxv=out.masked_fill(~mask.unsqueeze(-1),-1e9).amax(1)
        h=self.proj(torch.cat([mean,maxv],dim=-1))
        return [head(h) for head in self.heads]


def build(config):
    return TorchTextABSA("bilstm", lambda vocab,cfg: BiLSTMNetwork(vocab,cfg), config)
