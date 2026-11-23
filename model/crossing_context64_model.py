"""Experimental frozen steering CNN plus a full-frame 64x64 context head."""
import torch
from torch import nn
from model import ULTCNN

class CrossingContext64CNN(nn.Module):
    def __init__(self,threshold=.95):
        super().__init__();self.base=ULTCNN(6);self.context_head=nn.Linear(6*16*16,1)
        self.threshold=float(threshold)
    def context_features(self,x):return self.base.features(x).flatten(1)
    def forward(self,x):
        # Channel 0 repeats the original 16x16 values in 4x4 blocks.
        legacy=self.base(x[:,0:1,::4,::4].contiguous())
        gate=self.context_head(self.context_features(x[:,1:2].contiguous())).flatten().sigmoid()
        self.last_context_probability=gate.detach()
        stop=torch.where(gate>=self.threshold,legacy.max(1).values+2,legacy[:,5])
        return torch.cat([legacy[:,:5],stop[:,None]],1)
