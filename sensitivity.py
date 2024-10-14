import os
import torch
import torch.nn as nn
from modules.svd_linear import SVDLinear
from evaluate_utils import evaluate_model, evaluate_perplexity
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def calib_sensitivity_ppl(model, calib_loader, args, use_cache=True, lm_head=True):
    model.eval()
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if lm_head and "lm_head" in name:
                continue
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    if args.compress_kv_cache:
        param_ratio_candidates = [0.1 * i for i in range(1, 20)]
    else:
        param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}
        raw_linear.is_calibration_stage = True
        for param_ratio in param_ratio_candidates:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                rank_align=args.rank_align,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.sensitivity_samples)
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
            pbar.update(1)
        raw_linear.is_calibration_stage = False
        raw_linear.cached_svd = None
        setattr(info["father"], info["name"], raw_linear)
    return sensitivity_dict
