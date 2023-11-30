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



def enum_configs(model, tokenizer, sensitivity_dict, args):
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = open(
        f"{path}/enum_a{args.act_aware}_s{args.test_split}_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}.json",
        "a+",
    )

    module_dict={name:module for name, module in model.named_modules()}
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
    low = high//2
    for mid in range(high, low, -args.interval):
        layers_min_ratio={layername:1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername]=min(layers_min_ratio[layername], ratio)
        tot_params=0
        compress_params=0
        for layername, ratio in layers_min_ratio.items():
            # set ratio
            raw_linear = module_dict[layername]
            info = linear_info[raw_linear]
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=ratio,
                alpha=args.alpha,
                act_aware=args.act_aware,
                oc_split=args.test_split if raw_linear.in_features<raw_linear.out_features else 1,
                ic_split=args.test_split if raw_linear.in_features>raw_linear.out_features else 1,
            )
            setattr(info["father"], info["name"], svd_linear)
            tot_params+=raw_linear.weight.numel()
            compress_params+=raw_linear.weight.numel()*ratio
        result = evaluate_model(
            model,
            tokenizer,
            args.model_id,
            "",
            eval_ppl="wikitext2,ptb",
            limit=-1,
        )
        # nan test
        if result["wikitext2"] != result["wikitext2"]:
            break
        param_ratio=compress_params/tot_params
        msg=f"mid={mid}, param_ratio={param_ratio}"
        print(msg)
        print(result)
        log_file.write(f"{msg}\n")
        log_file.write(f"{result}\n")
        log_file.flush()


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

    cablib_dataset = "wikitext2"
    calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, 256)
    calib_input_distribution(model, calib_loader, args.scaling_method)
    sensitivity = calib_sensitivity(model, tokenizer,args)
    # calib_input_output_distribution(model, calib_loader)
    # train_input_output_scale(model, calib_loader)
    # calib_full_input(model, calib_loader)
    print_gpu_memory("before convert_to_svd_linear")
    enum_configs(model, tokenizer, sensitivity, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd",
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
        default='wikitext2',
        choices=['wikitext2', 'c4', 'ptb'],
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
        "--interval",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    main(args)