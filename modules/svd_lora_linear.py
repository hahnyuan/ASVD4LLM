import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDLoRALinear(nn.Module):
    def __init__(self, U, S, V, bias=None) -> None:
        super().__init__()
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(S)
        self.V = nn.Parameter(V)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear, r_ratio: float):
        rank = int(min(linear.weight.size()) * r_ratio)
        U, S, V = torch.svd_lowrank(linear.weight.data, q=rank)
        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # print shapes
        # print(f"U: {U.size()}, S: {S.size()}, V: {V.size()}, bias: {bias.size()}")
        return SVDLoRALinear(U, S, V, bias)

    def forward(self, x):
        # compute USV^Tx + b
        x = F.linear(x, self.V.t(), bias=None)
        x = x * self.S
        x = F.linear(x, self.U, bias=None)
        if self.bias is not None:
            x = x + self.bias
        return x
