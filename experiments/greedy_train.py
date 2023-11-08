import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from utils import print_gpu_memory
from datautils import get_calib_data, sample_train_loaders
import tqdm
from svd_init_utils import calib_input_distribution

class StopInferenceSignal(Exception):
    pass

train_loader=None
def train_layer(model,tokenizer,svd_linear, raw_linear):
    global train_loader
    if train_loader is None:
        train_loader=sample_train_loaders("wikitext2",tokenizer, 128, 3,seqlen=1024)
    optimizer=torch.optim.Adam(svd_linear.parameters(),lr=1e-5)
    def hook(m,i,o):
        raw_o=raw_linear(i[0]).detach()
        m.loss=F.mse_loss(o,raw_o)
        raise StopInferenceSignal()
    svd_linear.register_forward_hook(hook)
    def pre_hook(m,i):
        return (i[0].detach(),)
    svd_linear.register_forward_pre_hook(pre_hook)
    for epochi in range(1):
        for i,batch in enumerate(train_loader):
            batch.to(model.device)
            try:
                model(batch)
            except StopInferenceSignal:
                pass
            optimizer.zero_grad()
            svd_linear.loss.backward(retain_graph=True)
            if (i+1)%16==0 or i==len(train_loader)-1:
                print(svd_linear.loss.item())
                optimizer.step()
    # remove hook
    svd_linear._forward_hooks.clear()
    svd_linear._forward_pre_hooks.clear()


def convert_linear_to_svd_lora_linear(model, tokenizer, args):
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file=open(f"{path}/greedy_train_{args.act_aware}.json","a+")

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
    layers_config=[]
    
    for full_name,raw_linear,father,name in linear_dict:
        low=0
        high=1
        select_svd_linear=None
        log_file.write(full_name+'\n')
        ppl_target=args.ppl_target_st+(args.ppl_target_ed-args.ppl_target_st)*len(layers_config)/len(linear_dict)
        for i in range(4):
            ratio=(low+high)/2
            svd_linear = SVDLoRALinear.from_linear(
                    raw_linear,
                    n_param_ratio=ratio,
                    train_ratio=1,
                    lora_method=args.lora_method,
                    act_aware=args.act_aware,
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
            if result["wikitext2"]>ppl_target:
                low=ratio
            else:
                high=ratio
            
            # train
            train_layer(model,tokenizer,svd_linear, raw_linear)
            result_train = evaluate_model(
                model,
                tokenizer,
                args.model_id,
                "",
                eval_ppl="wikitext2",
                limit=15,
            )

            if result_train["wikitext2"]>ppl_target and result["wikitext2"]>ppl_target:
                low=ratio
            else:
                high=ratio
                if result_train["wikitext2"]<result["wikitext2"]:
                    select_svd_linear=svd_linear
                else:
                    select_svd_linear = SVDLoRALinear.from_linear(
                        raw_linear,
                        n_param_ratio=ratio,
                        lora_method=args.lora_method,
                        act_aware=args.act_aware,
                    )
            log_file.write(str({"ratio": ratio,"wikitext2":result["wikitext2"], "wikitext2 train":result_train["wikitext2"]}) + ",\n")
            log_file.flush()

        layers_config.append((full_name,high))
        log_file.write(str({"ratio_trace": [ratio for _,ratio in layers_config],"ppl_target":ppl_target}) + ",\n")
        
        if high==1:
            setattr(father, name, raw_linear)
        else:
            setattr(father, name, select_svd_linear)


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
        "--lora_method",
        type=str,
        default="UV",
        help="lora method, default: UV",
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
        help="use act aware svd lora",
    )
    args = parser.parse_args()

    

    main(args)
