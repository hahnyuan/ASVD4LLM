import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrainScaleLinear(nn.Module):
    def __init__(self,weight,bias):
        super().__init__()
        self.weight=weight
        if bias is not None:
            self.bias=bias
        else:
            self.bias=None
        self.Si=nn.Parameter(torch.zeros(1,weight.size(1)))
        self.So=nn.Parameter(torch.zeros(weight.size(0),1))

        self.is_scale=True
        

    @staticmethod
    def from_linear(
        linear: nn.Linear,
    ):
        new_linear=TrainScaleLinear(linear.weight,linear.bias)
        return new_linear.to(linear.weight.device)

    def forward(self, inp):
        if self.is_scale:
            # Si=F.softmax(self.Si.float(),dim=1)
            # So=F.softmax(self.So.float(),dim=0)
            # w=self.weight.float()*So*Si
            Si=F.sigmoid(self.Si)
            So=F.sigmoid(self.So)
            w=self.weight*So*Si
            delta=torch.randn_like(w)*(w.abs().mean()*0.1)
            w=w+delta.detach()
            w=w/So/Si
            w=w.to(self.weight.dtype)
        else:
            w=self.weight
        y=F.linear(inp,w,self.bias)
        return y

