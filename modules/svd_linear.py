import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV") -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
        self.truncation_rank = S.size(0)
        if sigma_fuse == "UV":
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        ic_split=1,
        oc_split=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        if hasattr(linear, "transform_mat") and linear.transform_mat.ndim == 2:
            W = linear.weight.data

            truncated_rank = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))

            if getattr(linear, "is_calibration_stage", False) and hasattr(linear, "cached_svd"):
                U, S, VT, transform_mat_inv = linear.cached_svd
            else:
                W = W.float()
                transform_mat = linear.transform_mat.to(W.device)
                try:
                    transform_mat_inv = torch.linalg.inv(transform_mat)
                except Exception as e:
                    print("Warning: scaling_diag_matrix is not full rank!")
                    transform_mat += 1e-6 * torch.eye(transform_mat.shape[0], device=W.device)
                    transform_mat_inv = torch.linalg.inv(transform_mat)
                transform_mat = transform_mat.float()
                transform_mat_inv = transform_mat_inv.float()
                transformed_W = torch.matmul(W, transform_mat)
                U, S, VT = torch.linalg.svd(transformed_W, full_matrices=False)
                if getattr(linear, "is_calibration_stage", False):
                    linear.cached_svd = (U, S, VT, transform_mat_inv)
            S = S[:truncated_rank]
            U = U[:, :truncated_rank]
            V = torch.matmul(VT[:truncated_rank, :], transform_mat_inv).T

            new_linear = SVDLinear(U, S, V, linear.bias, sigma_fuse)
            new_linear.to(linear.weight.dtype)
            if not getattr(linear, "is_calibration_stage", False):
                linear.transform_mat.to("cpu")
            return new_linear
        elif hasattr(linear, "transform_mat") and linear.transform_mat.ndim == 1:
            n_params = linear.weight.numel()
            compressed_params = int(n_params * param_ratio)
            assert ic_split == 1 or oc_split == 1
            rank = compressed_params // (linear.in_features + linear.out_features)
            # rank align
            rank = int(np.ceil(rank / rank_align) * rank_align)

            # print("rank", rank)
            w = linear.weight.data.float()
            transform_mat = linear.transform_mat + 1e-6  # avoid zero division
            w = w * transform_mat.view(1, -1)
            try:
                U, S, V = torch.svd_lowrank(w, q=rank)
            except:
                print(f"WARNING: svd failed for {linear}, disable svd")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
            V = V / transform_mat.view(-1, 1)

            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None
            new_linear = SVDLinear(U, S, V, bias, sigma_fuse)
            new_linear.to(linear.weight.dtype)
            return new_linear
        else:
            print("WARNING: Cannot find transform_mat")
            return linear

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y
