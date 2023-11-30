import os
import torch
import torch.nn as nn
from modules.svd_linear import SVDLinear
from evaluate import evaluate_model

@torch.no_grad()
def calib_sensitivity(model, tokenizer, args):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_s{args.test_split}_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}.pt"
    if os.path.exists(cache_file) and not args.disable_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

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

    sensitivity_dict = {}
    param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=args.act_aware,
                oc_split=args.test_split
                if raw_linear.in_features < raw_linear.out_features
                else 1,
                ic_split=args.test_split
                if raw_linear.in_features > raw_linear.out_features
                else 1,
            )
            setattr(info["father"], info["name"], svd_linear)
            result = evaluate_model(
                model,
                tokenizer,
                args.model_id,
                "",
                eval_ppl="wikitext2",
                limit=args.n_calib_samples,
            )
            ppl = result["wikitext2"]
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict