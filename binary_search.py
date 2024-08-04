import os
import torch
import torch.nn as nn
from evaluate_utils import evaluate_model, evaluate_perplexity
from modules.svd_linear import SVDLinear, GradSVDLinear
from tqdm import tqdm


def binary_search_truncation_rank(model, sensitivity_dict, calib_loader, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]

                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    if args.compress_kv_cache:
        # after compression, we have 2 matrices, so kv_cache reduction is the twice of weight reduction
        param_ratio_target = args.kv_cache_ratio_target * 2
        sensitivity_dict = {k: v for k, v in sensitivity_dict.items() if "k_proj" in k or "v_proj" in k}
        assert args.ppl_target < 0, "ppl_target is not supported when compressing kv_cache"
    else:
        param_ratio_target = args.param_ratio_target

    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            if not args.compress_kv_cache and ratio >= 1:
                # we need to compress the weights, so parameter ratio should be less than 1
                continue
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    assert args.ppl_target > 0 or param_ratio_target > 0

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                    rank_align=args.rank_align,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}({compress_params}/{tot_params})"
            print(msg)
            if param_ratio > param_ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"Searching finished, decomposing layers...")
    layers_min_ratio = {layername: None for layername in sensitivity_dict.keys()}
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        if layers_min_ratio[layername] is None:
            layers_min_ratio[layername] = ratio
        else:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
    for layername, ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        if ratio is None:
            svd_linear = raw_linear
        else:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=ratio,
                alpha=args.alpha,
                act_aware=args.act_aware,
                sigma_fuse=args.sigma_fuse,
                rank_align=args.rank_align,
            )
        setattr(info["father"], info["name"], svd_linear)


def binary_search_truncation_rank_optimize_scale(model, sensitivity_dict, calib_loader, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    assert args.ppl_target > 0 or args.param_ratio_target > 0

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = GradSVDLinear.from_linear(
                    raw_linear,
                    param_ratio=ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}({compress_params}/{tot_params})"
            print(msg)
            if param_ratio > args.param_ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"Searching finished, decomposing layers...")
    layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
    for layername, ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        svd_linear = GradSVDLinear.from_linear(
            raw_linear,
            param_ratio=ratio,
            alpha=args.alpha,
            act_aware=args.act_aware,
            sigma_fuse=args.sigma_fuse,
        )
        setattr(info["father"], info["name"], svd_linear)
