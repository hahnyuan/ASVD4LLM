from modules.svd_lora_linear import SVDLoRALinear
import torch.nn as nn
import torch
import torch.nn.functional as F


# unittest
def test_SVDLoRALinear():
    x = torch.randn(10, 128)
    linear = nn.Linear(128, 64)
    linear.input_abs_mean = torch.mean(torch.abs(x), dim=0)
    
    # print(linear.weight.size())
    for r_ratio in torch.linspace(0.1, 1, 10):
        svd_linear = SVDLoRALinear.from_linear(linear, r_ratio, act_aware=True)
        x = torch.randn(1, 128)
        y = linear(x)
        y_svd = svd_linear(x)
        diff = torch.norm(y - y_svd)
        print(f"r_ratio={r_ratio:.2f}, diff={diff:.2f}")
    y=linear(x)
    linear.output_abs_mean=torch.mean(torch.abs(y),dim=0)
    # print(linear.weight.size())
    for r_ratio in torch.linspace(0.1, 1, 10):
        svd_linear = SVDLoRALinear.from_linear(linear, r_ratio, act_aware=True)
        x = torch.randn(1, 128)
        y = linear(x)
        y_svd = svd_linear(x)
        diff = torch.norm(y - y_svd)
        print(f"2d r_ratio={r_ratio:.2f}, diff={diff:.2f}")


if __name__ == "__main__":
    test_SVDLoRALinear()
