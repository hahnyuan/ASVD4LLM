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
from modules.act_aware_svd_lora_linear import ActAwareSVDLoRALinear
from utils import print_gpu_memory
from datautils import get_calib_data


def calib_input_distribution(model, calib_loader):
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
        module.input_abs_mean += abs_mean

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.input_abs_mean = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in calib_loader:
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)


def convert_linear_to_svd_lora_linear(model, tokenizer, args):
    full_name_dict = {module: name for name, module in model.named_modules()}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, child in submodule.named_children():
            if isinstance(child, nn.Linear):
                full_name = full_name_dict[child]
                for rank_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    svd_linear = SVDLoRALinear.from_linear(
                        child,
                        r_ratio=rank_ratio,
                        lora_method=args.lora_method,
                        act_aware=args.act_aware,
                    )
                    print(f"convert {full_name} to svd_lora_linear ratio={rank_ratio}")
                    setattr(submodule, name, svd_linear)
                    result = evaluate_model(
                        model,
                        tokenizer,
                        args.model_id,
                        "boolq",
                        eval_ppl="wikitext2",
                        limit=200,
                    )
                    del result["boolq"]
                    result.update({"rank_ratio": rank_ratio, "full_name": full_name})
                    with open(
                        f"output/sensitivity_{args.model_id.replace('/','_')}_{args.act_aware}.json",
                        "a+",
                    ) as f:
                        f.write(str(result) + ",\n")
                setattr(submodule, name, child)

            else:
                modules.append(child)


def total_model_parameters_buffers(model):
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.buffers()
    )


def main(args):
    # Dataset
    # data_name = "mlabonne/guanaco-llama2-1k"
    # data_name = 'SirNeural/flan_v2'
    # data_name = 'databricks/databricks-dolly-15k'

    # Model and tokenizer names
    # base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    model_id = args.model_id
    # "./checkpoints/opt-1.3b-lora-mlabonne-enhanced-svd"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    raw_model_parameters, raw_model_buffers = total_model_parameters_buffers(model)
    print("raw model tot: {}".format(raw_model_parameters + raw_model_buffers))
    if args.act_aware:
        cablib_dataset = "wikitext2"
        calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, 256)
        calib_input_distribution(model, calib_loader)
    print_gpu_memory("before convert_linear_to_svd_lora_linear")
    convert_linear_to_svd_lora_linear(model, tokenizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--msa_rank_ratio",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--mlp_rank_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lora_method",
        type=str,
        default="UV",
        help="lora method, default: UV",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd lora",
    )
    args = parser.parse_args()

    main(args)
