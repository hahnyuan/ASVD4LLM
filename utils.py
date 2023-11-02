import torch


def print_gpu_memory(prefix=""):
    print(
        f"{prefix} GPU memory allocated: {torch.cuda.memory_allocated()}, max: {torch.cuda.max_memory_allocated()}"
    )
