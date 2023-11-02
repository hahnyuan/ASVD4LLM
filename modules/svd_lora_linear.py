import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDLoRALinear(nn.Module):
    def __init__(
        self, U, S, V, bias=None, train_ratio=0.5, lora_method="Uonly"
    ) -> None:
        super().__init__()
        rank = U.size(1)
        train_rank = int(rank * train_ratio)
        if lora_method == "Uonly":
            u_rank = 0
            v_rank = rank
        elif lora_method == "Vonly":
            u_rank = rank
            v_rank = 0
        elif lora_method == "UV":
            u_rank = train_rank
            v_rank = train_rank
        else:
            raise ValueError(f"lora_method {lora_method} not supported")
        self.U_fix = U[:, :u_rank]
        self.U_train = nn.Parameter(U[:, u_rank:])
        self.V_fix = V[:, :v_rank]
        self.V_train = nn.Parameter(V[:, v_rank:])
        self.S = S
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @staticmethod
    def from_linear(
        linear: nn.Linear, r_ratio: float, train_ratio: float = 0.5, lora_method="Uonly"
    ):
        rank = int(min(linear.weight.size()) * r_ratio)
        U, S, V = torch.svd_lowrank(linear.weight.data, q=rank)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # print shapes
        # print(f"U: {U.size()}, S: {S.size()}, V: {V.size()}, bias: {bias.size()}")
        return SVDLoRALinear(U, S, V, bias, train_ratio, lora_method)

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
