from modules.svd_lora_linear import SVDLoRALinear
from torch import nn
import torch


class ActAwareSVDLoRALinear(SVDLoRALinear):
    @staticmethod
    def from_linear(
        linear: nn.Linear, r_ratio: float, train_ratio: float = 0.5, lora_method="Uonly"
    ):
        scaling_diag_matrix = linear.scaling_diag_matrix
        scaling_diag_matrix += 1e-6  # avoid zero division
        rank = int(min(linear.weight.size()) * r_ratio)
        U, S, V = torch.svd_lowrank(linear.weight.data * scaling_diag_matrix, q=rank)

        V = V / scaling_diag_matrix.view(-1, 1)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        # check inf nan
        inf_check = torch.isinf(U).any() or torch.isinf(S).any() or torch.isinf(V).any()
        nan_check = torch.isnan(U).any() or torch.isnan(S).any() or torch.isnan(V).any()
        if inf_check or nan_check:
            breakpoint()  # debug
        assert not inf_check and not nan_check, "inf or nan detected"

        # print(f"U: {U.size()}, S: {S.size()}, V: {V.size()}, bias: {bias.size()}")
        return ActAwareSVDLoRALinear(U, S, V, bias, train_ratio, lora_method)
