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
        act_sensitivity = mlp[-1].input_abs_mean
        indexes = torch.argsort(act_sensitivity)
        reorder_indexes = indexes.view(-1, args.test_split).transpose(0, 1).reshape(-1)
        # reorder output
        for layer in mlp[:-1]:
            layer.weight.data = layer.weight.data[reorder_indexes]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[reorder_indexes]
        # reorder input
        mlp[-1].weight.data = mlp[-1].weight.data[:, reorder_indexes]
        mlp[-1].input_abs_mean = mlp[-1].input_abs_mean[reorder_indexes]
        print(f"reorder for {full_name_dict[mlp[-1]]} done")


def convert_to_svd_linear(model, tokenizer, args):
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = open(
        f"{path}/greedy_split_a{args.act_aware}_r{args.reorder}_s{args.test_split}.json",
        "a+",
    )

    full_name_dict = {module: name for name, module in model.named_modules()}
    if "opt" in args.model_id:
        self_attns = [
            (
                _.self_attn.q_proj,
                _.self_attn.k_proj,
                _.self_attn.v_proj,
                _.self_attn.out_proj,
            )
            for _ in model.model.decoder.layers
        ]
        mlps = [(_.fc1, _.fc2) for _ in model.model.decoder.layers]
    elif "llama" in args.model_id:
        self_attns = [
            (_.attn.q_proj, _.attn.k_proj, _.attn.v_proj, _.attn.out_proj)
            for _ in model.model.layers
        ]
        mlps = [
            (_.mlp.gate_proj, _.mlp.up_proj, _.mlp.down_proj)
            for _ in model.model.layers
        ]
    else:
        raise NotImplementedError

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

    # multi-binary search
    ratio_trace=[]
    raw_params=0
    compressed_params=0
    for layeri, mlp in enumerate(mlps):
        ppl_target = args.ppl_target_st + (args.ppl_target_ed - args.ppl_target_st) * (
            layeri + 1
        ) / len(mlps)
        delta_ratio=[0.25 for _ in mlp]
        now_ratio=[0.5 for _ in mlp]
        for fci,fc in enumerate(mlp):
            raw_params += fc.weight.numel()
            ratio=now_ratio[fci]
            svd_linear = SVDLinear.from_linear(
                fc,
                param_ratio=ratio,
                act_aware=args.act_aware,
                oc_split=args.test_split if fc.in_features < fc.out_features else 1,
                ic_split=args.test_split if fc.in_features > fc.out_features else 1,
            )
            setattr(linear_info[fc]["father"], linear_info[fc]["name"], svd_linear)
        result = evaluate_model(
            model,
            tokenizer,
            args.model_id,
            "",
            eval_ppl="wikitext2",
            limit=15,
        )
        ppl = result["wikitext2"]
        if ppl != ppl:
            breakpoint()
            ppl = 1e10
        prev_ppl=ppl
        is_ratio_up = ppl > ppl_target
        # select the fc that makes ppl improved most
        # binary search
        for i in range(8):
            best_fci=None
            best_delta_ppl=None
            for fci,fc in enumerate(mlp):
                ratio=now_ratio[fci]
                ratio+=delta_ratio[fci] if is_ratio_up else -delta_ratio[fci]
                svd_linear = SVDLinear.from_linear(
                    fc,
                    param_ratio=ratio,
                    act_aware=args.act_aware,
                    oc_split=args.test_split if fc.in_features < fc.out_features else 1,
                    ic_split=args.test_split if fc.in_features > fc.out_features else 1,
                )
                prev_svd_linear=getattr(linear_info[fc]["father"], linear_info[fc]["name"])
                setattr(linear_info[fc]["father"], linear_info[fc]["name"], svd_linear)
                result = evaluate_model(
                    model,
                    tokenizer,
                    args.model_id,
                    "",
                    eval_ppl="wikitext2",
                    limit=15,
                )
                ppl = result["wikitext2"]
                if ppl != ppl:
                    breakpoint()
                    ppl = 1e10
                delta_ppl=ppl-prev_ppl
                if best_delta_ppl is None or delta_ppl>best_delta_ppl:
                    best_delta_ppl=delta_ppl
                    best_fci=fci
                setattr(linear_info[fc]["father"], linear_info[fc]["name"], prev_svd_linear)
                log_file.write(f"{fci} {'up' if is_ratio_up else 'down'} {prev_ppl}+{delta_ppl}={ppl}\n")
                log_file.flush()
            now_ratio[best_fci]+=delta_ratio[best_fci] if is_ratio_up else -delta_ratio[best_fci]
            prev_ppl+=best_delta_ppl
            is_ratio_up = prev_ppl > ppl_target
            delta_ratio[best_fci]/=2
            fc=mlp[best_fci]
            ratio=now_ratio[best_fci]
            svd_linear = SVDLinear.from_linear(
                fc,
                param_ratio=ratio,
                act_aware=args.act_aware,
                oc_split=args.test_split if fc.in_features < fc.out_features else 1,
                ic_split=args.test_split if fc.in_features > fc.out_features else 1,
            )
            setattr(linear_info[fc]["father"], linear_info[fc]["name"], svd_linear)
            log_file.write(f"== change {best_fci} {'up' if is_ratio_up else 'down'} {now_ratio}\n")
        for fci,fc in enumerate(mlp):
            ratio=now_ratio[fci]
            ratio_trace.append(ratio)
            new_fc=getattr(linear_info[fc]["father"], linear_info[fc]["name"])
            for U, S, V in zip(
                new_fc.Us, new_fc.Ss, new_fc.Vs
            ):
                compressed_params += U.numel() +  V.numel()
        log_file.write(
            f"{ratio_trace}\n ppl_target {ppl_target} - now_compression_ratio {compressed_params/raw_params}\n"
        )
        print(
            f"now_comp_ratio {compressed_params/raw_params} ppl_target {ppl_target} - now_ppl {prev_ppl}"
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
        "--reorder",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
