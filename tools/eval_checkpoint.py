import argparse
import os
import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from evaluate import evaluate_model
from svd_lora_train import (
    convert_to_svd_linear,
    total_model_parameters_buffers,
)


def main(args):
    # Dataset
    # data_name = "mlabonne/guanaco-llama2-1k"
    # data_name = 'SirNeural/flan_v2'
    # data_name = 'databricks/databricks-dolly-15k'

    # Model and tokenizer names
    # base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    model_id = args.model_id

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    # convert_to_svd_linear(model, args)
    if args.path:
        if os.path.exists(args.path + "/pytorch_model.bin"):
            state_dict = torch.load(args.path + "/pytorch_model.bin", map_location="cuda:0")
            model.load_state_dict(state_dict)
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.path)
            model = model.merge_and_unload()
            model.save_pretrained(os.path.join(args.path, "final_merged"))
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(args.path, "final_merged")
            )

    model.eval()

    svd_model_parameters, svd_model_buffers = total_model_parameters_buffers(model)
    print("svd model tot: {}".format(svd_model_parameters + svd_model_buffers))

    model_merged = model
    query = "### Human: I am depressed, what should I do?"
    text_gen = pipeline(
        task="text-generation",
        model=model_merged,
        tokenizer=tokenizer,
        max_length=200,
    )
    # output = text_gen(f"<s>[INST] {query} [/INST]")
    output = text_gen(query)
    print(output[0]["generated_text"])
    # evaluate_model(base_model, llama_tokenizer, base_model_name, 'llmqat', limit=200, eval_ppl=False)
    results = evaluate_model(
        model_merged,
        tokenizer,
        model_id,
        "llmqat",
        limit=args.limit,
        eval_ppl="wikitext2,ptb,c4",
        num_fewshot=0,
    )
    # save results to txt
    with open(args.path + "/eval_results.txt", "w") as f:
        f.write(str(results))
    results = evaluate_model(
        model_merged,
        tokenizer,
        model_id,
        "mmlu",
        limit=args.limit,
        eval_ppl=False,
        num_fewshot=0,
    )
    # save results to txt
    with open(args.path + "/eval_results.txt", "w") as f:
        f.write(str(results))
    results = evaluate_model(
        model_merged,
        tokenizer,
        model_id,
        "mmlu",
        limit=args.limit,
        eval_ppl=False,
        num_fewshot=5,
    )
    # save results to txt
    with open(args.path + "/eval_results.txt", "w") as f:
        f.write(str(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
    )
    parser.add_argument(
        "--path",
        type=str,
        help="checkpoint path",
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
    parser.add_argument("--limit", type=int, default=200, help="limit of eval data")
    args = parser.parse_args()

    main(args)
