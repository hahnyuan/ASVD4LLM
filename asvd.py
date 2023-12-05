import argparse
import os
import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


from evaluate import evaluate_model,evaluate_perplexity
from modules.svd_linear import SVDLinear

from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import calib_sensitivity
from quantization import rtn_quant_sequential


def search_best_compression_ratio(model, tokenizer, sensitivity_dict, calib_loader, args):
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = open(
        f"{path}/ssearch_a{args.act_aware}_{args.scaling_method}_{args.alpha}.json",
        "a+",
    )

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
    assert args.ppl_target>0 or args.param_ratio_target>0
    
    input_ids=torch.cat([_['input_ids'] for _ in calib_loader],0)
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0
        for layername, ratio in layers_min_ratio.items():
            # set ratio
            raw_linear = module_dict[layername]
            info = linear_info[raw_linear]
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=ratio,
                alpha=args.alpha,
                act_aware=args.act_aware,
            )
            setattr(info["father"], info["name"], svd_linear)
            tot_params += raw_linear.weight.numel()
            compress_params += raw_linear.weight.numel() * ratio
        ppl = evaluate_perplexity(model,input_ids, args.n_calib_samples)
        param_ratio = compress_params / tot_params
        msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
        print(msg)
        log_file.write(f"{msg}\n")
        if args.ppl_target>0:
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            if param_ratio > args.param_ratio_target:
                high = mid
            else:
                low = mid + 1


def main(args):
    model_id = args.model_id

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )

    model = model.to_bettertransformer()

    # sensitivity calibration
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, 256)
    if args.scaling_method == "fisher":
        calib_fisher_info(model, calib_loader, args.use_cache)
    else:
        calib_input_distribution(
            model, calib_loader, args.scaling_method, args.use_cache
        )
    sensitivity = calib_sensitivity(model, tokenizer, calib_loader, args, args.use_cache)

    # search best compression ratio
    search_best_compression_ratio(model, tokenizer, sensitivity, calib_loader, args)

    # quantization
    if args.weight_quant != "none":
        if args.weight_quant=="rtn_int8":
            rtn_quant_sequential(model, 8)
        elif args.weight_quant=="rtn_int6":
            rtn_quant_sequential(model, 6)

    # evaluate
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "",
        eval_ppl="wikitext2,ptb",
        limit=-1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher"],
        help="scaling method",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6"],
        help="weight quantization method",
    )
    args = parser.parse_args()

    main(args)
