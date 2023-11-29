import argparse
import os
import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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

from utils import print_gpu_memory
from datautils import get_calib_data, sample_train_loaders
from tqdm import tqdm
from svd_init_utils import calib_input_distribution, calib_input_output_distribution
import numpy as np


def inf_nan_trace(model):
    def hook(module, input, output):
        if len(input) and isinstance(input[0], torch.Tensor):
            if torch.isnan(input[0]).any():
                breakpoint()
            if torch.isinf(input[0]).any():
                breakpoint()
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                breakpoint()
            if torch.isinf(output).any():
                breakpoint()

    for name, module in model.named_modules():
        module._forward_hooks.clear()
        module.register_forward_hook(hook)


def reorder_mlp(model):
    full_name_dict = {module: name for name, module in model.named_modules()}
    if "opt" in args.model_id:
        mlps = [(_.fc1, _.fc2) for _ in model.model.decoder.layers]
    elif "llama" in args.model_id:
        mlps = [
            (_.mlp.gate_proj, _.mlp.up_proj, _.mlp.down_proj)
            for _ in model.model.layers
        ]
    else:
        raise NotImplementedError
    if args.test_split == 1:
        return
    for mlp in mlps:
        act_sensitivity = mlp[-1].scaling_diag_matrix
        indexes = torch.argsort(act_sensitivity)
        reorder_indexes = indexes.view(-1, args.test_split).transpose(0, 1).reshape(-1)
        # reorder output
        for layer in mlp[:-1]:
            layer.weight.data = layer.weight.data[reorder_indexes]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[reorder_indexes]
        # reorder input
        mlp[-1].weight.data = mlp[-1].weight.data[:, reorder_indexes]
        mlp[-1].scaling_diag_matrix = mlp[-1].scaling_diag_matrix[reorder_indexes]
        print(f"reorder for {full_name_dict[mlp[-1]]} done")


def convert_to_svd_linear(model, tokenizer, args):
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = open(
        f"{path}/greedy_it{args.n_round}_split_a{args.act_aware}_r{args.reorder}_s{args.test_split}_co{args.cosearch}.json",
        "a+",
    )

    full_name_dict = {module: name for name, module in model.named_modules()}
    all_linears = []

    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                all_linears.append(raw_linear)
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    # binary searchq
    for roundi in range(args.n_round):
        split_trace = []
        ratio_trace = []
        raw_params = 0
        compressed_params = 0
        for layeri, raw_linear in enumerate(all_linears[::-1]):
            # individual search
            now_progress = (roundi * len(all_linears) + layeri) / (
                len(all_linears) * args.n_round
            )
            ppl_target = (
                args.ppl_target_st
                + (args.ppl_target_ed - args.ppl_target_st) * now_progress
            )
            msg = f"round {roundi} mlp {full_name_dict[raw_linear]}, ppl_target={ppl_target}\n"
            log_file.write(msg)
            print(msg)
            full_name = linear_info[raw_linear]["full_name"]
            father = linear_info[raw_linear]["father"]
            name = linear_info[raw_linear]["name"]
            log_file.write(f"individual search {full_name}\n")
            low = 0
            high = 1

            best_split = None
            best_svd_linear = None
            if raw_linear.in_features > raw_linear.out_features:
                # split_candidates=((1,1),(args.test_split,1))
                split_candidates = ((args.test_split, 1),)
            elif raw_linear.in_features < raw_linear.out_features:
                # split_candidates=((1,1),(1,args.test_split))
                split_candidates = ((1, args.test_split),)
            else:
                split_candidates = ((1, 1),)
            for i in range(4):
                ratio = (low + high) / 2
                min_ppl = 1e10
                min_split = None
                min_svd_linear = None
                for ic_split, oc_split in split_candidates:
                    try:
                        svd_linear = SVDLinear.from_linear(
                            raw_linear,
                            param_ratio=ratio,
                            act_aware=args.act_aware,
                            oc_split=oc_split,
                            ic_split=ic_split,
                        )
                    except:
                        continue
                    setattr(father, name, svd_linear)
                    # inf_nan_trace(model)
                    result = evaluate_model(
                        model,
                        tokenizer,
                        args.model_id,
                        "",
                        eval_ppl="wikitext2",
                        limit=15,
                    )
                    ppl = result["wikitext2"]
                    # nan check
                    if ppl != ppl:
                        # breakpoint()
                        ppl = 1e10
                    if min_ppl is None or ppl < min_ppl:
                        min_ppl = ppl
                        min_split = (ic_split, oc_split)
                        min_svd_linear = svd_linear

                    log_file.write(
                        str(
                            {
                                "ic/oc split": (ic_split, oc_split),
                                "ratio": ratio,
                                "wikitext2": result["wikitext2"],
                            }
                        )
                        + ",\n"
                    )
                    log_file.flush()
                if min_ppl > ppl_target:
                    low = ratio
                else:
                    high = ratio
                    best_split = min_split
                    best_svd_linear = min_svd_linear

            split_trace.append(best_split)
            ratio_trace.append(high)
            raw_params += raw_linear.weight.numel()
            print(f"{full_name} high={high} low={low}")
            if high == 1:
                setattr(father, name, raw_linear)
                compressed_params += raw_linear.weight.numel()
            else:
                setattr(father, name, best_svd_linear)
                for U, S, V in zip(
                    best_svd_linear.Us, best_svd_linear.Ss, best_svd_linear.Vs
                ):
                    compressed_params += U.numel() + S.numel() + V.numel()
            log_file.write(
                f"{split_trace}\n{ratio_trace}\n ppl_target {ppl_target} min_ppl {min_ppl} - now_compression_ratio {compressed_params/raw_params}\n"
            )
            print(
                f"now_comp_ratio {compressed_params/raw_params} ppl_target {ppl_target} - now_ppl {min_ppl}"
            )


def total_model_parameters_buffers(model):
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.buffers()
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

    raw_model_parameters, raw_model_buffers = total_model_parameters_buffers(model)
    print("raw model tot: {}".format(raw_model_parameters + raw_model_buffers))
    # if args.act_aware:
    cablib_dataset = "wikitext2"
    calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, 256)
    # calib_input_distribution(model, calib_loader)
    calib_input_output_distribution(model, calib_loader)
    print_gpu_memory("before convert_to_svd_linear")
    if args.reorder:
        reorder_mlp(model)
    convert_to_svd_linear(model, tokenizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ppl_target_st",
        type=float,
    )
    parser.add_argument(
        "--ppl_target_ed",
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
        "--n_round",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--reorder",
        action="store_true",
    )
    parser.add_argument(
        "--cosearch",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
