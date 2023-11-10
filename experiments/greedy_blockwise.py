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

from utils import print_gpu_memory
from datautils import get_calib_data, sample_train_loaders
import tqdm
from svd_init_utils import calib_input_distribution


def convert_to_svd_linear(model, tokenizer, args):
    path = f"output/{args.model_id.replace('/','_')}"
    if not os.path.exists(path):
        os.makedirs(path)
    log_file=open(f"{path}/greedy_{args.act_aware}.json","a+")

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
        log_file.write(full_name+'\n')
        for i in range(4):
            ratio=(low+high)/2
            svd_linear = SVDLoRALinear.from_linear(
                    raw_linear,
                    param_ratio=ratio,
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
            ppl_target=args.ppl_target_st+(args.ppl_target_ed-args.ppl_target_st)*len(layers_config)/len(linear_dict)
            if result["wikitext2"]>ppl_target:
                low=ratio
            else:
                high=ratio
            log_file.write(str({"ratio": ratio,"wikitext2":result["wikitext2"]}) + ",\n")
            log_file.flush()
        layers_config.append((full_name,high))
        log_file.write(str({"ratio_trace": [ratio for _,ratio in layers_config],"ppl_target":ppl_target}) + ",\n")
        
        if high==1:
            setattr(father, name, raw_linear)
        else:
            svd_linear = SVDLoRALinear.from_linear(
                    raw_linear,
                    param_ratio=ratio,
                    lora_method=args.lora_method,
                    act_aware=args.act_aware,
                )
            setattr(father, name, svd_linear)
        

    # compression_ratios=[1,0.8,0.6,0.4,0.2,0.1]
    # config_stack=[]
    # for ratio in compression_ratios:
    #     full_name,raw_linear,submodule,name=linear_dict[0]
    #     config_stack.append([(full_name, ratio)])
    # while len(config_stack) > 0:
    #     layers_config = config_stack.pop()
    #     for i, (full_name, ratio) in enumerate(layers_config):
    #         if ratio==1:
    #             full_name,raw_linear,submodule,name=linear_dict[i]
    #             setattr(submodule, name, raw_linear)
    #         else:
    #             full_name,raw_linear,submodule,name=linear_dict[i]
    #             svd_linear = SVDLoRALinear.from_linear(
    #                 raw_linear,
    #                 compression_ratio=ratio,
    #                 lora_method=args.lora_method,
    #                 act_aware=args.act_aware,
    #             )
    #             setattr(submodule, name, svd_linear)
    #             # print(
    #             #     f"convert {full_name} to svd_lora_linear ratio={ratio}"
    #             # )
            
    #     for i in range(len(layers_config),len(linear_dict)):
    #         full_name,raw_linear,submodule,name=linear_dict[i]
    #         setattr(submodule, name, raw_linear)
    #         # print(f"convert {full_name} to raw_linear")
    #     result = evaluate_model(
    #         model,
    #         tokenizer,
    #         args.model_id,
    #         "",
    #         eval_ppl="wikitext2",
    #         limit=15,
    #     )
    #     ratio_trace=[ratio for _,ratio in layers_config]
    #     # calculate ppl_target by linear interpolation
    #     ppl_target=args.ppl_target_st+(args.ppl_target_ed-args.ppl_target_st)*len(layers_config)/len(linear_dict)
    #     print(result["wikitext2"],ppl_target,ratio_trace)
    #     if 0:
    #         for i in range(len(linear_dict)):
    #             full_name,raw_linear,submodule,name=linear_dict[i]
    #             setattr(submodule, name, raw_linear)
    #         result2 = evaluate_model(
    #             model,
    #             tokenizer,
    #             args.model_id,
    #             "",
    #             eval_ppl="wikitext2",
    #             limit=50,
    #         )
    #         print(result2["wikitext2"])
        
    #     log_file.write(str({"ratio_trace": ratio_trace,"wikitext2":result["wikitext2"],"ppl_target":ppl_target}) + ",\n")
    #     log_file.flush()
    #     if result["wikitext2"]>ppl_target:
    #         continue
    #     else:
    #         full_name2,_,_,_=linear_dict[len(layers_config)]
    #         for ratio in compression_ratios:
    #             config_stack.append(layers_config+[(full_name2, ratio)])

        


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
    print_gpu_memory("before convert_to_svd_linear")
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
