"""Experimental shared CNN features for cropped steering and full-frame context.

The original six-class CNN is frozen. An additional small binary head sees
both feature maps. No position, track map, or line tracker is an input.
FPGA export of this experimental paired-input architecture is not provided.
"""
import torch
from torch import nn
from model import ULTCNN

class CrossingContextCNN(nn.Module):
    def __init__(self,threshold=0.95):
        super().__init__();self.base=ULTCNN(6)
        self.context_head=nn.Sequential(nn.Linear(192,8),nn.ReLU(),nn.Linear(8,1))
        self.threshold=float(threshold)

    def context_features(self,x):
        cropped=self.base.features(x[:,0:1]).flatten(1)
        full=self.base.features(x[:,1:2]).flatten(1)
        return torch.cat([cropped,full],1)

    def forward(self,x):
        features=self.context_features(x)
        legacy=self.base.classifier(features[:,:96].reshape(-1,6,4,4))
        gate=self.context_head(features).flatten().sigmoid()
        stop=torch.where(gate>=self.threshold,legacy.max(1).values+2,legacy[:,5])
        return torch.cat([legacy[:,:5],stop[:,None]],1)
