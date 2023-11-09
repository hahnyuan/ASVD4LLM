import torch
import torch.nn as nn
import torch.nn.functional as F


class PruneLinear(nn.Module):
    def __init__(
        self, weight, bias=None
    ) -> None:
        super().__init__()
        self.weight=nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        n_param_ratio: float,
        prune_dim=0,
    ):
        n_prune=int(linear.weight.size(prune_dim)*(1-n_param_ratio))
        if prune_dim==0:
            # argsort by output_std
            output_std = linear.output_std
            _, indices=torch.sort(output_std)
            indices=indices[:n_prune]
            output_mean=linear.output_mean
            new_weight=linear.weight.clone()
            new_weight[indices]=0
            if linear.bias is not None:
                new_bias=linear.bias.clone()
            else:
                new_bias=torch.zeros_like(output_mean)
            new_bias[indices]=output_mean[indices]
            # breakpoint()
        elif prune_dim==1:
            # argsort by input_std
            input_std = linear.input_std
            _, indices=torch.sort(input_std)
            indices=indices[:n_prune]
            input_mean=linear.input_mean
            new_weight=linear.weight.clone()
            new_weight[:,indices]=0

            bias_shift=(linear.weight[:,indices]*input_mean[indices].view(1,-1)).sum(dim=1)
            if linear.bias is not None:
                new_bias=linear.bias.clone()
            else:
                new_bias=torch.zeros_like(output_mean)
            new_bias+=bias_shift
        else:
            raise ValueError("prune_dim must be 0 or 1")
        return PruneLinear(new_weight, new_bias)
          

    def forward(self, x):
        y=F.linear(x,self.weight, self.bias)
        return y

