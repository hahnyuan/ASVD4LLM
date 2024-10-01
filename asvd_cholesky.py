import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, LlamaTokenizer
from transformers.models.opt.configuration_opt import OPTConfig
from evaluate_utils import evaluate_model
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info, layerwise_cholesky_decomposition
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from quantization import rtn_quant_sequential, awq_quant_sequential
from binary_search import binary_search_truncation_rank
import numpy as np


def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load model
    model_id = args.model_id
    print(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    # generate transform_mat
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, args.n_calib_samples, seed=args.seed)

    cache_dir = f"cache/{args.model_id.replace('.', '_').replace('/', '_')}"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    transform_mat_cache = (
        f"{cache_dir}/{args.transform_mat_method}_{args.calib_dataset}_{args.n_calib_samples}_{args.seed}.pt"
    )
    if args.use_cache and os.path.exists(transform_mat_cache):
        print(f"Loading transform matrix from CACHE: {transform_mat_cache}")
        transform_mat = torch.load(transform_mat_cache, map_location="cpu")
    else:
        model.seqlen = 2048
        if "magnitude" in args.transform_mat_method:
            # TODO: **alpha
            raise NotImplementedError("alpha not set, is not implemented")
            transform_mat = calib_input_distribution(model, calib_loader, args.transform_mat_method, args.use_cache)
        elif "cholesky" in args.transform_mat_method:
            transform_mat = layerwise_cholesky_decomposition(args.model_id, model, calib_loader, args.DEV)
        torch.save(transform_mat, transform_mat_cache)
    if "opt" in model_id:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        for name, module in layer.named_modules():
            if name in transform_mat[i]:
                # print(f"Profiling matrix found for {name}")
                module.transform_mat = transform_mat[i][name]

    # evaluate sensitivity for each linear layer
    model.to("cuda")
    sensitivity_cache = f"{cache_dir}/{args.transform_mat_method}_{args.sensitivity_metric}_{args.calib_dataset}_{args.n_calib_samples}_{args.seed}.pt"
    if args.use_cache and os.path.exists(sensitivity_cache):
        print(f"Loading sensitivity from CACHE: {sensitivity_cache}")
        sensitivity = torch.load(sensitivity_cache, map_location="cpu")
    else:
        if args.sensitivity_metric == "ppl":
            sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
        # elif args.sensitivity_metric == "stable_rank":
        #     sensitivity = calib_sensitivity_stable_rank(model, calib_loader, args, args.use_cache)
        else:
            raise NotImplementedError
        torch.save(sensitivity, sensitivity_cache)

    # Sensitivity-based Truncation Rank Searching (STRS)
    binary_search_truncation_rank(model, sensitivity, calib_loader, args)

    # quantization (optional)
    if args.weight_quant != "none":
        if args.weight_quant == "rtn_int8":
            rtn_quant_sequential(model, 8)
        elif args.weight_quant == "rtn_int6":
            rtn_quant_sequential(model, 6)
        elif args.weight_quant == "awq_int8":
            model = awq_quant_sequential(model, tokenizer, 8)
        elif args.weight_quant == "awq_int4":
            model = awq_quant_sequential(model, tokenizer, 4)

    # evaluate the compressed model
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
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
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "alpaca", "selfgen"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--transform_mat_method",
        type=str,
        default="cholesky",
        choices=["magnitude_alpha0.5", "cholesky"],
        help="transform matrix setting method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank"],
        help="search metric",
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
