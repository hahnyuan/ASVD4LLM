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
        self.lora_method = lora_method
        if lora_method == "Uonly":
            u_rank = 0
            v_rank = rank
        elif lora_method == "Vonly":
            u_rank = rank
            v_rank = 0
        elif lora_method == "UV":
            u_rank = rank-train_rank
            v_rank = rank-train_rank 
        elif lora_method == "UVall":
            u_rank = 0
            v_rank = 0
        else:
            raise ValueError(f"lora_method {lora_method} not supported")
        U=U*S.sqrt()
        V=V*S.sqrt()
        U_fix = U[:, :u_rank]
        self.register_buffer("U_fix", U_fix)
        self.U_train = nn.Parameter(U[:, u_rank:])
        V_fix = V[:, :v_rank]
        self.register_buffer("V_fix", V_fix)
        self.V_train = nn.Parameter(V[:, v_rank:])
        self.S = S
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        train_ratio: float = 0.5,
        lora_method="Uonly",
        act_aware=False,
    ):
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        # compressed_params=rank*(in_features+out_features)
        # therefore, rank=compressed_params/(in_features+out_features)
        rank = compressed_params // (linear.in_features + linear.out_features)
        print(f"rank: {rank}")
        # rank = int(min(linear.weight.size()) * compression_ratio)
        if act_aware:
            scaling_diag_matrix = linear.scaling_diag_matrix
            scaling_diag_matrix += 1e-6  # avoid zero division
            if hasattr(linear, "output_abs_mean"):
                output_abs_mean = linear.output_abs_mean
                output_abs_mean += 1e-6  # avoid zero division
                scaling_diag_matrix = scaling_diag_matrix.sqrt()
                output_abs_mean = output_abs_mean.sqrt()
                w = (
                    linear.weight.data
                    * output_abs_mean.view(-1, 1)
                    * scaling_diag_matrix.view(1, -1)
                )
            else:
                w = linear.weight.data * scaling_diag_matrix.view(1, -1)
        else:
            w = linear.weight.data
        U, S, V = torch.svd_lowrank(w, q=rank)
        # print(S)
        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)
            if hasattr(linear, "output_abs_mean"):
                U = U / output_abs_mean.view(-1, 1)
        if lora_method == "reconstruct":
            w_approx = torch.matmul(
                U, torch.matmul(S.diag_embed(), V.transpose(-2, -1))
            )
            new_layer = nn.Linear(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
            )
            new_layer.weight.data = w_approx
            if linear.bias is not None:
                new_layer.bias.data = linear.bias.data
            return new_layer

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
        # x = x * self.S
        x = F.linear(x, U, bias=None)
        if self.bias is not None:
            x = x + self.bias
        return x

    def rebuild_weight(self):
        V = torch.cat([self.V_fix, self.V_train], dim=1)
        U = torch.cat([self.U_fix, self.U_train], dim=1)
        weight = torch.matmul(U, torch.matmul(self.S.diag_embed(), V.transpose(-2, -1)))
        return weight
