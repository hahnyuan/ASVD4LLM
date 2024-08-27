import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, LlamaTokenizer
from transformers.models.opt.configuration_opt import OPTConfig
from evaluate_utils import evaluate_model
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank, calib_sensitivity_ppl_greedy
from quantization import rtn_quant_sequential, awq_quant_sequential
from binary_search import binary_search_truncation_rank
import numpy as np
from modelutils import profile_svdllm_low_resource


def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load model
    model_id = args.model_id
    print(model_id)
    if "opt" in model_id or "mistral" in model_id or "Qwen2" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # if "llama" in model_id or "opt" in model_id:
    #     model = model.to_bettertransformer()

    # sensitivity calibration
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, args.n_calib_samples, seed=args.seed)
    # if "fisher" in args.scaling_method:
    #     calib_fisher_info(model, calib_loader, args.use_cache)
    # if "abs" in args.scaling_method:
    #     calib_input_distribution(model, calib_loader, args.scaling_method, args.use_cache)

    if args.profiling_path is not None:
        profiling_mat = torch.load(args.profiling_path, map_location="cpu")
    else:
        model.seqlen = 2048
        profiling_mat = profile_svdllm_low_resource(args.model_id, model, calib_loader, args.DEV)
        if args.save_path is not None:
            torch.save(
                profiling_mat,
                args.save_path
                + "/"
                + args.model_id.replace("/", "_").replace("-", "_")
                + "_profiling_"
                + args.calib_dataset
                + "_"
                + str(args.seed)
                + ".pt",
            )
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        for name, module in layer.named_modules():
            if name in profiling_mat[i]:
                # print(f"Profiling matrix found for {name}")
                whitening_matrix = profiling_mat[i][name]
                module.whitening_matrix = whitening_matrix

    if args.sensitivity_metric == "ppl":
        sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
    elif args.sensitivity_metric == "stable_rank":
        sensitivity = calib_sensitivity_stable_rank(model, calib_loader, args, args.use_cache)
    elif args.sensitivity_metric == "greedy_ppl":
        sensitivity = calib_sensitivity_ppl_greedy(model, calib_loader, args, args.use_cache)

    # search best truncation rank for each layer

    binary_search_truncation_rank(model, sensitivity, calib_loader, args)

    # quantization
    if args.weight_quant != "none":
        if args.weight_quant == "rtn_int8":
            rtn_quant_sequential(model, 8)
        elif args.weight_quant == "rtn_int6":
            rtn_quant_sequential(model, 6)
        elif args.weight_quant == "awq_int8":
            model = awq_quant_sequential(model, tokenizer, 8)
        elif args.weight_quant == "awq_int4":
            model = awq_quant_sequential(model, tokenizer, 4)

    # evaluate
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "mmlu" if args.eval_mmlu else args.eval_tasks,
        eval_ppl=args.eval_ppl,
        limit=-1,
    )
    print(result)
    if not os.path.exists("output"):
        os.makedirs("output")
    with open("output/result.txt", "a+") as f:
        f.write(f"{args}\n")
        f.write(f"{result}\n")

    # finished


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--profiling_path",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="c4",
        choices=["wikitext2", "c4", "ptb", "alpaca", "selfgen"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="whitening",
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="greedy_ppl",
        choices=["ppl", "stable_rank", "greedy_ppl"],
        help="search metric",
    )
    parser.add_argument(
        "--greedy_thres",
        type=float,
        default="0.02",
        help="threshold for greedy search", 
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6", "awq_int8", "awq_int4"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--eval_ppl",
        default="wikitext2,ptb",
        type=str,
    )
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed, which can significantly affect the calibration results",
    )
    parser.add_argument(
        "--compress_kv_cache",
        action="store_true",
        help="compress kv cache by asvd for k_proj and v_proj",
    )
    parser.add_argument(
        "--kv_cache_ratio_target",
        type=float,
        default=-1,
        help="kv cache ratio",
    )
    parser.add_argument(
        "--rank_align",
        type=int,
        default=1,
        help="align rank in SVD",
    )
    parser.add_argument("--DEV", type=str, default="cuda", help="device")
    parser.add_argument("--model_seq_len", type=int, default=2048, help="the default sequence length of the LLM")
    args = parser.parse_args()

    main(args)
