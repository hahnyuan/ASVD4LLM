import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None) -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)
        self.ALinear.weight.data = U.mul(S.sqrt())
        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
        self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1))

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        reorder=False,
        gradient_aware=False,
        ic_split=1,
        oc_split=1,
        train_scale=False,
        act_full=False,
        alpha=1,
        scale_dim=1,
    ):
        if param_ratio >= 1:
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = linear.scaling_diag_matrix**alpha
            scaling_diag_matrix += 1e-6  # avoid zero division
            assert scale_dim==1
            if scale_dim==1:
                w = w * scaling_diag_matrix.view(1, -1)
            elif scale_dim==0:
                w = w * scaling_diag_matrix.view(-1, 1)
            else:
                raise NotImplementedError
        Us = []
        Ss = []
        Vs = []
        try:
            U, S, V = torch.svd_lowrank(w, q=rank)
        except:
            print(f"svd failed for {linear}, disable act_aware")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
            )
        if act_aware:
            if scale_dim==0:
                S = S / scaling_diag_matrix.view(-1, 1)
            elif scale_dim==1:
                V = V / scaling_diag_matrix.view(-1, 1)
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan check
        for S in Ss:
            if torch.isnan(S).any():
                print("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )
        for U in Us:
            if torch.isnan(U).any():
                print("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )
        for V in Vs:
            if torch.isnan(V).any():
                print("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias)
        return new_linear.to(linear.weight.dtype)

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y
