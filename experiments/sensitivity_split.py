import argparse
import os
import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    OPTPreTrainedModel,
    TrainingArguments,
    pipeline,
)

from trl import SFTTrainer

# from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset

from evaluate import evaluate_model
from modules.svd_lora_linear import SVDLoRALinear
from modules.svd_linear import SVDLinear
from modules.train_scale_linear import TrainScaleLinear

from utils import print_gpu_memory
from datautils import get_calib_data, sample_train_loaders
from tqdm import tqdm
from svd_init_utils import calib_input_distribution, calib_input_output_distribution
import torch.nn.functional as F


@torch.no_grad()
def calib_sensitivity(model, tokenizer):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity.pt"
    if os.path.exists(cache_file):
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
                # act_full=True,
                act_aware=args.act_aware,
                # oc_split=oc_split,
                # ic_split=ic_split,
            )
            setattr(info["father"], info["name"], svd_linear)
            result = evaluate_model(
                model,
                tokenizer,
                args.model_id,
                "",
                eval_ppl="wikitext2",
                limit=15,
            )
            ppl = result["wikitext2"]
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
        setattr(info["father"], info["name"], raw_linear)
        torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


def search_best_compression_ratio(model, tokenizer, sensitivity_dict, args):
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = open(
        f"{path}/ssearch_a{args.act_aware}_s{args.test_split}.json",
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
    low = 0
    ppl_target = args.ppl_target
    while low < high:
        mid = (low + high) // 2
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
                # act_full=True,
                act_aware=args.act_aware,
                # oc_split=oc_split,
                # ic_split=ic_split,
            )
            setattr(info["father"], info["name"], svd_linear)
            tot_params+=raw_linear.weight.numel()
            compress_params+=raw_linear.weight.numel()*ratio
        result = evaluate_model(
            model,
            tokenizer,
            args.model_id,
            "",
            eval_ppl="wikitext2",
            limit=15,
        )
        ppl = result["wikitext2"]
        param_ratio=compress_params/tot_params
        print(f"mid={mid}, ppl={ppl}, param_ratio={param_ratio}")
        log_file.write(f"mid={mid}, ppl={ppl}, param_ratio={param_ratio}\n")
        if ppl < ppl_target:
            high = mid
        else:
            low = mid + 1
        print(f"Update: high={high}, low={low}")
    mid = (low + high) // 2
    layers_min_ratio={layername:1 for layername, ratio, ppl in sorted_sensitive_list[:mid+1]}
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername]=min(layers_min_ratio[layername], ratio)
    for layername, ratio in layers_min_ratio.items():
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        svd_linear = SVDLinear.from_linear(
            raw_linear,
            param_ratio=ratio,
            # act_full=True,
            act_aware=args.act_aware,
            oc_split=args.test_split if raw_linear.in_features<raw_linear.out_features else 1,
            ic_split=args.test_split if raw_linear.in_features>raw_linear.out_features else 1,
        )
        setattr(info["father"], info["name"], svd_linear)
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "",
        eval_ppl="wikitext2",
        limit=15,
    )
    ppl = result["wikitext2"]
    print(f"mid={mid}, ppl={ppl}")
    log_file.write(
        str(
            {
                "mid": mid,
                "wikitext2": result["wikitext2"],
            }
        )
        + ",\n"
    )




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
    calib_input_distribution(model, calib_loader)
    sensitivity = calib_sensitivity(model, tokenizer)
    # calib_input_output_distribution(model, calib_loader)
    # train_input_output_scale(model, calib_loader)
    # calib_full_input(model, calib_loader)
    print_gpu_memory("before convert_to_svd_linear")
    search_best_compression_ratio(model, tokenizer, sensitivity, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd",
    )
    parser.add_argument(
        "--test_split",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--reorder",
        action="store_true",
    )
    parser.add_argument(
        "--cosearch",
        action="store_true",
    )
    parser.add_argument(
        "--load_scale",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
