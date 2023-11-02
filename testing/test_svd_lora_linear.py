from modules.svd_lora_linear import SVDLoRALinear
import torch.nn as nn
import torch
import torch.nn.functional as F


# unittest
def test_SVDLoRALinear():
    linear = nn.Linear(128, 64)
    # print(linear.weight.size())
    for r_ratio in torch.linspace(0.1, 1, 10):
        svd_linear = SVDLoRALinear.from_linear(linear, r_ratio)
        x = torch.randn(1, 128)
        y = linear(x)
        y_svd = svd_linear(x)
        diff = torch.norm(y - y_svd)
        print(f"r_ratio={r_ratio:.2f}, diff={diff:.2f}")


if __name__ == "__main__":
    test_SVDLoRALinear()
