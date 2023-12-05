import argparse
import os
import torch
import torch.nn as nn

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
import matplotlib.pyplot as plt
import numpy as np
import re
        

def run_eval(model, tokenizer, sensitivity_dict, args, save_path):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                try:
                    block_name = int(re.findall(r"\.(\d+)\.", full_name)[0])
                except:
                    block_name='lm_head'
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                    "sub_name": name.split(".")[-1],
                    "block_name":block_name
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    # for act_aware in [False, True]:
    for act_aware in [True]:
        all_layer_type_ratio={}
        all_block_ratio={}
        for target_params_ratio in np.linspace(0.75, 1, 10):
            layer_type_ratio={}
            block_ratio={}
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
                sub_name = info["sub_name"]
                block_name = info["block_name"]
                params=raw_linear.weight.numel()
                if sub_name not in layer_type_ratio:
                    layer_type_ratio[sub_name]=[]
                layer_type_ratio[sub_name].append(ratio)
                if block_name not in block_ratio:
                    block_ratio[block_name]=[]
                block_ratio[block_name].append((ratio,params))
            all_layer_type_ratio[target_params_ratio]=layer_type_ratio
            all_block_ratio[target_params_ratio]=block_ratio
        torch.save((all_block_ratio,all_layer_type_ratio), f"{save_path}/analysis_{args.model_id.replace('/','_')}.pt")
        # plot layer_type_ratio with bar plot, x-axis is target ratio, y-axis is ratio of each layer type
        


def main(args):
    model_id = args.model_id

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )

    model = model.to_bettertransformer()
    save_path = f"output/final/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

    cablib_dataset = "wikitext2"
    calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, 256)
    calib_input_distribution(model, calib_loader, args.scaling_method)
    sensitivity = calib_sensitivity(model, tokenizer, args)
    # calib_input_output_distribution(model, calib_loader)
    # train_input_output_scale(model, calib_loader)
    # calib_full_input(model, calib_loader)
    print_gpu_memory("before convert_to_svd_linear")
    run_eval(model, tokenizer, sensitivity, args, save_path)


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
        "--mmlu",
        action="store_true",
    )
    parser.add_argument(
        "--original_naive",
        action="store_true",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    main(args)
