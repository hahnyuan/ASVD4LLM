import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDLoRALinear(nn.Module):
    def __init__(self, U, S, V, bias=None, train_ratio=0.5) -> None:
        super().__init__()
        rank = U.size(1)
        train_rank = int(rank * train_ratio)
        self.U_fix = U[:, :train_rank]
        self.U_train = nn.Parameter(U[:, train_rank:])
        self.V_fix = V[:, :train_rank]
        self.V_train = nn.Parameter(V[:, train_rank:])
        self.S = S
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear, r_ratio: float, train_ratio: float = 0.5):
        rank = int(min(linear.weight.size()) * r_ratio)
        U, S, V = torch.svd_lowrank(linear.weight.data, q=rank)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # print shapes
        # print(f"U: {U.size()}, S: {S.size()}, V: {V.size()}, bias: {bias.size()}")
        return SVDLoRALinear(U, S, V, bias, train_ratio)

    def forward(self, x):
        # compute USV^Tx + b
        V = torch.cat([self.V_fix, self.V_train], dim=1)
        U = torch.cat([self.U_fix, self.U_train], dim=1)
        x = F.linear(x, V.t(), bias=None)
        x = x * self.S
        x = F.linear(x, U, bias=None)
        if self.bias is not None:
            x = x + self.bias
        return x
