from modules.svd_lora_linear import SVDLoRALinear
from modules.svd_linear import SVDLinear
import torch.nn as nn
import torch
import torch.nn.functional as F


# unittest
def test_SVDLinear():
    linear = nn.Linear(128, 64)
    # print(linear.weight.size())
    x = torch.randn(10, 128)
    for r_ratio in torch.linspace(0.1, 1, 10):
        svd_linear = SVDLinear.from_linear(linear, r_ratio,ic_split=2)
        y = linear(x)
        y_svd = svd_linear(x)
        diff = torch.norm(y - y_svd)
        print(f"r_ratio={r_ratio:.2f}, diff={diff:.2f}")
    for r_ratio in torch.linspace(0.1, 1, 10):
        svd_linear = SVDLinear.from_linear(linear, r_ratio,oc_split=2)
        y = linear(x)
        y_svd = svd_linear(x)
        diff = torch.norm(y - y_svd)
        print(f"r_ratio={r_ratio:.2f}, diff={diff:.2f}")
    for r_ratio in torch.linspace(0.1, 1, 10):
        svd_linear = SVDLoRALinear.from_linear(linear, r_ratio)
        y = linear(x)
        y_svd = svd_linear(x)
        diff = torch.norm(y - y_svd)
        print(f"svd_lora r_ratio={r_ratio:.2f}, diff={diff:.2f}")


if __name__ == "__main__":
    test_SVDLinear()
