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
from datautils import get_calib_data, sample_train_loaders
import tqdm
from svd_init_utils import calib_input_distribution


def convert_linear_to_svd_lora_linear(model, tokenizer, args):
    full_name_dict = {module: name for name, module in model.named_modules()}

    linear_dict=[]
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_dict.append((full_name,raw_linear,submodule,name))
            else:
                modules.append(raw_linear)
    
    ratios=eval(args.ratios) # a list of ratios
    for i, ratio in enumerate(ratios):
        full_name,raw_linear,submodule,name=linear_dict[i]
        if ratio==1:
            setattr(submodule, name, raw_linear)
        else:
            svd_linear = SVDLoRALinear.from_linear(
                raw_linear,
                compression_ratio=ratio,
                lora_method=args.lora_method,
                act_aware=args.act_aware,
            )
            setattr(submodule, name, svd_linear)
                # print(
                #     f"convert {full_name} to svd_lora_linear ratio={ratio}"
                # )
            
    for i in range(len(ratios),len(linear_dict)):
        full_name,raw_linear,submodule,name=linear_dict[i]
        setattr(submodule, name, raw_linear)
            # print(f"convert {full_name} to raw_linear")
    


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

    model = model.to_bettertransformer()

    raw_model_parameters, raw_model_buffers = total_model_parameters_buffers(model)
    print("raw model tot: {}".format(raw_model_parameters + raw_model_buffers))
    if args.act_aware:
        cablib_dataset = "wikitext2"
        calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, 256)
        calib_input_distribution(model, calib_loader)
    if args.ratios:
        convert_linear_to_svd_lora_linear(model, tokenizer, args)
    model_parameters, model_buffers = total_model_parameters_buffers(model)
    print("model tot: {}".format(model_parameters + model_buffers))
    # print fraction
    print(
        "param fraction: {}%".format(
            (model_parameters + model_buffers) / (raw_model_parameters + raw_model_buffers)
        )
    )
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "llmqat",
        eval_ppl="wikitext2",
        limit=200,
    )
    log_file.write(args.ratios + ",\n")
    log_file.write(str(result) + ",\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
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
    parser.add_argument("--ratios",type=str)
    args = parser.parse_args()
    
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file_name=f"{path}/greedy_{args.act_aware}_result.json"
    log_file=open(log_file_name,"a+")

    main(args)

    
