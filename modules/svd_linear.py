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
        act_aware=False,
        ic_split=1,
        oc_split=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        if hasattr(linear, "transform_mat"):
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

            # truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            # sqrtSigma = torch.sqrt(truc_sigma)
            # svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            # svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)

            new_linear = SVDLinear(U, S, V, linear.bias, sigma_fuse)
            new_linear.to(linear.weight.dtype)
            if not getattr(linear, "is_calibration_stage", False):
                linear.transform_mat.to("cpu")
            return new_linear
        elif hasattr(linear, "scaling_diag_matrix") or hasattr(linear, "fisher_info"):
            raise NotImplementedError
            # if param_ratio >= 1:
            #     return linear
            n_params = linear.weight.numel()
            compressed_params = int(n_params * param_ratio)
            assert ic_split == 1 or oc_split == 1
            rank = compressed_params // (linear.in_features + linear.out_features)
            # rank align
            rank = int(np.ceil(rank / rank_align) * rank_align)

            # print("rank", rank)
            w = linear.weight.data.float()
            if act_aware:
                transform_mat = 1  # avoid zero division
                if hasattr(linear, "scaling_diag_matrix"):
                    # print("WARNING: scaling_diag_matrix is used")
                    transform_mat *= linear.scaling_diag_matrix**alpha
                    # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
                if hasattr(linear, "fisher_info"):
                    transform_mat *= linear.fisher_info**alpha
                    # scaling_diag_matrix *= linear.fisher_info**1
                # if not (scaling_diag_matrix == scaling_diag_matrix).all():
                #     breakpoint()
                transform_mat += 1e-6  # avoid zero division
                w = w * transform_mat.view(1, -1)
            Us = []
            Ss = []
            Vs = []
            try:
                U, S, V = torch.svd_lowrank(w, q=rank)
            except:
                print(f"svd failed for {linear}, disable act_aware")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
            if act_aware:
                V = V / transform_mat.view(-1, 1)
            Us = [U]
            Ss = [S]
            Vs = [V]

            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None

            # nan or inf check
            for S in Ss:
                if (S != S).any():
                    print("nan in S")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for U in Us:
                if (U != U).any():
                    print("nan in U")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )
            for V in Vs:
                if (V != V).any():
                    print("nan in V")
                    return (
                        nn.Linear(linear.in_features, linear.out_features)
                        .to(linear.weight.dtype)
                        .to(linear.weight.device)
                    )

            assert len(Us) == len(Ss) == len(Vs) == 1
            new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse)
            new_linear.to(linear.weight.dtype)
            return new_linear
        else:
            print("Cannot find scaling_diag_matrix or fisher_info, disable act_aware")
            return linear

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        # y=torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.ALinear(y)
        # y=torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y


class WhiteningSVDLinear(SVDLinear):
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        W = linear.weight.data.float()
        # dtype = W.dtype
        scaling_diag_matrix = linear.scaling_diag_matrix
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        except Exception as e:
            print("Warning: scaling_diag_matrix is not full rank!")
            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0], device=W.device)
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = scaling_matrix_inv.float()
        W_scale = torch.matmul(W, scaling_diag_matrix)
        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))
        truc_s = S[:num_s_after_trunc]
        truc_u = U[:, :num_s_after_trunc]
        truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
        # truc_sigma = torch.diag(truc_s)
        #### Replace Attn, MLP ####
        # sqrtSigma = torch.sqrt(truc_sigma)
        # svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
        # svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
        new_linear = SVDLinear(truc_u, truc_s, truc_v, linear.bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        return new_linear


class GradSVDLinear(nn.Module):
    def __init__(self, weight, scale, bias, rank) -> None:
        super().__init__()
        self.weight = weight
        self.scale = nn.Parameter(scale)
        self.bias = bias
        self.rank = rank

    @staticmethod
    def from_linear(
        linear: nn.Linear, param_ratio: float, act_aware=False, ic_split=1, oc_split=1, alpha=1, sigma_fuse="UV"
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
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        return GradSVDLinear(w, scaling_diag_matrix, bias, rank)

    def forward(self, inp):
        w = self.weight * self.scale.view(1, -1)
        U, S, V = torch.svd_lowrank(w, q=self.rank)
        new_w = U.mul(S).mm(V.t())
        y = F.linear(inp, new_w, self.bias)
        return y
