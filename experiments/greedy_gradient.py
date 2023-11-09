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
import tqdm
from svd_init_utils import calib_input_distribution,calib_input_output_distribution


def convert_linear_to_svd_lora_linear(model, tokenizer, args):
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file=open(f"{path}/greedy_gradient_{args.gradient_aware}{args.reorder}.json","a+")

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
    log_file.write(str([_[0] for _ in linear_dict]) + ",\n")
    # binary search
    ratio_trace=[]
    raw_params=0
    compressed_params=0
    for layeri,(full_name,raw_linear,father,name) in enumerate(linear_dict):
        if 'head' in full_name:
            continue
        # if 'fc2' not in full_name:
        #     continue
        log_file.write(full_name+'\n')
        low=0
        high=1
        ppl_target=args.ppl_target_st+(args.ppl_target_ed-args.ppl_target_st)*(layeri+1)/len(linear_dict)
        best_svd_linear=None
        for i in range(4):
            ratio=(low+high)/2

            if 'self_attn' in full_name:
                svd_linear=SVDLinear.from_linear(
                        raw_linear,
                        param_ratio=ratio,
                        act_aware=args.act_aware,
                        reorder=args.reorder,
                )
            else:
                svd_linear=SVDLinear.from_linear(
                        raw_linear,
                        param_ratio=ratio,
                        gradient_aware=args.gradient_aware,
                        reorder=args.reorder,
                )
            setattr(father, name, svd_linear)
            result = evaluate_model(
                model,
                tokenizer,
                args.model_id,
                "",
                eval_ppl="wikitext2",
                limit=15,
            )
            ppl=result["wikitext2"]
                    
            log_file.write(str({"ratio": ratio,"wikitext2":result["wikitext2"]}) + ",\n")
            log_file.flush()
            if ppl>ppl_target:
                low=ratio
            else:
                high=ratio
                best_svd_linear=svd_linear
            
        ratio_trace.append(high)
        raw_params+=raw_linear.weight.numel()
        if high==1:
            setattr(father, name, raw_linear)
            compressed_params+=raw_linear.weight.numel()
        else:
            setattr(father, name, best_svd_linear)
            for U,S,V in zip(best_svd_linear.Us,best_svd_linear.Ss,best_svd_linear.Vs):
                compressed_params+=U.numel()+S.numel()+V.numel()
        log_file.write(
            f"{ratio_trace}\n ppl_target {ppl_target} - now_compression_ratio {compressed_params/raw_params}\n")

def get_gradient(model,trainloader):
    print(f"get_gradient")
    # hook for each Linear layer, to get collect the input activation gradient and output activation gradient
    def hook_fn(module, grad_input, grad_output):
        module.input_grad += grad_input[0].view(-1,grad_input[0].size(-1)).abs().mean(0)
        module.output_grad += grad_output[0].view(-1,grad_output[0].size(-1)).abs().mean(0)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.input_grad = 0
            module.output_grad = 0
            module.register_full_backward_hook(hook_fn)

    for batch in tqdm.tqdm(trainloader):
        batch = batch.to(model.device)
        model.zero_grad()
        outputs = model(batch)
        loss = outputs.loss
        loss['logits'].mean().backward()
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module._backward_hooks.clear()

        


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
    # if args.act_aware:
    cablib_dataset = "wikitext2"
    calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, 256)
    # calib_input_distribution(model, calib_loader)
    calib_input_output_distribution(model, calib_loader)
    train_loader=sample_train_loaders('wikitext2',tokenizer,16,3,1024)
    get_gradient(model,train_loader)
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
        "--ppl_target_st",
        type=float,
    )
    parser.add_argument(
        "--ppl_target_ed",
        type=float,
    )
    parser.add_argument(
        "--gradient_aware",
        action="store_true",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
    )
    parser.add_argument(
        "--reorder",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
