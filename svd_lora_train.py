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
from datasets import load_dataset

from evaluate import evaluate_model
from modules.svd_lora_linear import SVDLoRALinear
from utils import print_gpu_memory


def convert_linear_to_svd_lora_linear(module):
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Linear):
            if "lora" not in name:
                svd_linear = SVDLoRALinear.from_linear(
                    submodule, r_ratio=args.rank_compress_ratio
                )
                del submodule.weight
                setattr(module, name, svd_linear)
                # clear memory
                # print(f"convert {name} to svd_lora_linear")
        # elif isinstance(submodule, nn.Embedding):
        #     if "lora" not in name:
        #         svd_linear = SVDLoRALinear.from_linear(
        #             submodule, r_ratio=args.rank_compress_ratio
        #         )
        #         setattr(module, name, svd_linear)
        else:
            convert_linear_to_svd_lora_linear(submodule)


def total_model_parameters_buffers(model):
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.buffers()
    )


def train(model, tokenizer, train_dataset, args):
    # Training Params
    train_params = TrainingArguments(
        output_dir="./output/svd_lora_train",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # optim="paged_adamw_32bit",
        optim="adamw_torch",
        save_steps=1000,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="linear",
    )

    # Training
    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params,
    )

    # Training
    fine_tuning.train()


def main(args):
    # Dataset
    # data_name = "mlabonne/guanaco-llama2-1k"
    # data_name = 'SirNeural/flan_v2'
    # data_name = 'databricks/databricks-dolly-15k'

    # Model and tokenizer names
    # base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    model_id = args.model_id
    refined_model = args.saved_dir
    # "./checkpoints/opt-1.3b-lora-mlabonne-enhanced-svd"

    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    raw_model_parameters, raw_model_buffers = total_model_parameters_buffers(model)
    print("raw model tot: {}".format(raw_model_parameters + raw_model_buffers))
    print_gpu_memory("before convert_linear_to_svd_lora_linear")
    convert_linear_to_svd_lora_linear(model)
    torch.cuda.empty_cache()
    print_gpu_memory("after convert_linear_to_svd_lora_linear")

    svd_model_parameters, svd_model_buffers = total_model_parameters_buffers(model)
    print("svd model tot: {}".format(svd_model_parameters + svd_model_buffers))
    print(
        "tot ratio:{}".format(
            (svd_model_parameters + svd_model_buffers)
            / (raw_model_parameters + raw_model_buffers)
        )
    )
    print("train ratio: {}".format(svd_model_parameters / raw_model_parameters))

    # train
    if not args.inference_mode:
        # train_dataset = get_qat_dataset(training_data, llama_tokenizer, 256)
        data_name = "tatsu-lab/alpaca"
        # data_name = 'timdettmers/openassistant-guanaco'
        training_dataset = load_dataset(data_name, split="train")
        train(model, llama_tokenizer, training_dataset, args)

    model_merged = model
    query = "### Human: I am depressed, what should I do?"
    text_gen = pipeline(
        task="text-generation",
        model=model_merged,
        tokenizer=llama_tokenizer,
        max_length=200,
    )
    # output = text_gen(f"<s>[INST] {query} [/INST]")
    output = text_gen(query)
    print(output[0]["generated_text"])
    # evaluate_model(base_model, llama_tokenizer, base_model_name, 'llmqat', limit=200, eval_ppl=False)
    evaluate_model(
        model_merged,
        llama_tokenizer,
        model_id,
        "llmqat",
        limit=-1,
        eval_ppl=False,
        num_fewshot=0,
    )
    evaluate_model(
        model_merged,
        llama_tokenizer,
        model_id,
        "mmlu",
        limit=-1,
        eval_ppl=False,
        num_fewshot=0,
    )
    evaluate_model(
        model_merged,
        llama_tokenizer,
        model_id,
        "mmlu",
        limit=-1,
        eval_ppl=False,
        num_fewshot=5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--saved_dir",
        type=str,
        default="./checkpoints/opt-1.3b-lora-mlabonne-enhanced-svd",
        help="dir for saving the lora and merged weights",
    )
    parser.add_argument(
        "--using_svd",
        action="store_true",
        default=False,
        help="if using svd for pretrained weight",
    )
    parser.add_argument(
        "--rank_compress_ratio",
        type=float,
        default=0.2,
        help="for svd, default: 0.17",
    )
    parser.add_argument(
        "--inference_mode",
        action="store_true",
        default=False,
        help="inference only mode",
    )
    args = parser.parse_args()

    main(args)
