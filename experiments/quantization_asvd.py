import argparse
import os
import torch
import torch.nn as nn
import copy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from evaluate import evaluate_model
from modules.svd_linear import SVDLinear

from utils import print_gpu_memory
from datautils import get_calib_data, sample_train_loaders
from svd_init_utils import calib_input_distribution, calib_input_output_distribution
from sensitivity import calib_sensitivity
from quantization import gptq_quant_sequential,rtn_quant_sequential



def run_eval(model, tokenizer, sensitivity_dict, args, log_file, gptq_loader):
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
    act_aware=True
    for target_params_ratio in [0.95, 0.9, 0.85, 0.8, 0.75]:
        high = len(sorted_sensitive_list) - 1
        low = 0
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
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            param_ratio = compress_params / tot_params
            if param_ratio > target_params_ratio:
                high = mid
            else:
                low = mid + 1
        for layername, ratio in layers_min_ratio.items():
            # set ratio
            raw_linear = module_dict[layername]
            info = linear_info[raw_linear]
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=ratio,
                alpha=args.alpha,
                act_aware=act_aware,
                oc_split=args.test_split
                if raw_linear.in_features < raw_linear.out_features
                else 1,
                ic_split=args.test_split
                if raw_linear.in_features > raw_linear.out_features
                else 1,
            )
            setattr(info["father"], info["name"], svd_linear)
        device="cuda:0"
        qmodel=copy.deepcopy(model)
        if args.quant_method=='rtn':
            rtn_quant_sequential(qmodel, gptq_loader, device, args)
        elif args.quant_method=='gptq':
            gptq_quant_sequential(qmodel, gptq_loader, device, args)
        qmodel=qmodel.to(device)
        result = evaluate_model(
            qmodel,
            tokenizer,
            args.model_id,
            "",
            eval_ppl="wikitext2,ptb",
            limit=-1,
        )
        msg=f"{args.quant_method} {args.wbits}bit act-aware={act_aware} target_params_ratio={target_params_ratio}\n"
        print(msg)
        print(result)
        log_file.write(msg)
        log_file.write(str(result))
        log_file.flush()


def main(args):
    model_id = args.model_id

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float16
    )

    model = model.to_bettertransformer()

    save_path = f"output/final/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = open(f"{save_path}/{args.model_id.replace('/','_')}_quant.json", "a+")

    cablib_dataset = "wikitext2"
    calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, args.nsamples)
    calib_input_distribution(model, calib_loader, args.scaling_method)
    sensitivity = calib_sensitivity(model, tokenizer, args)
    # calib_input_output_distribution(model, calib_loader)
    # train_input_output_scale(model, calib_loader)
    # calib_full_input(model, calib_loader)
    print_gpu_memory("before convert_to_svd_linear")
    model.seqlen=2048
    run_eval(model, tokenizer, sensitivity, args, log_file, calib_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--test_split",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max"],
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--wbits', type=int, default=8,
    )
    parser.add_argument(
        '--groupsize', type=int, default=128,
    )
    parser.add_argument(
        '--quant_method', type=str, default='rtn', choices=['rtn', 'gptq']
    )
    args = parser.parse_args()

    main(args)
