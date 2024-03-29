import sys

sys.path.append(".")
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig
from evaluate_utils import evaluate_model
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from quantization import rtn_quant_sequential
from binary_search import binary_search_truncation_rank
from modules.svd_linear import SVDLinear
import os


def main(args):
    model_id = args.model_id

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # sensitivity calibration
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, 256)
    if "fisher" in args.scaling_method:
        calib_fisher_info(model, calib_loader, args.use_cache)
    if "abs" in args.scaling_method:
        calib_input_distribution(
            model, calib_loader, args.scaling_method, args.use_cache
        )
    if args.sensitivity_metric == "ppl":
        sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
    elif args.sensitivity_metric == "stable_rank":
        sensitivity = calib_sensitivity_stable_rank(
            model, calib_loader, args, args.use_cache
        )

    # search best truncation rank for each layer

    binary_search_truncation_rank(model, sensitivity, calib_loader, args)

    # build huggingface model
    assert args.param_ratio_target > 0
    assert args.act_aware
    assert args.alpha == 0.5
    assert args.calib_dataset == "wikitext2"
    assert args.scaling_method == "abs_mean"
    assert args.sensitivity_metric == "ppl"
    assert args.use_cache
    assert args.weight_quant == "none"
    assert not args.eval_mmlu

    save_path = (
        "huggingface_repos/"
        + model_id.split("/")[-1]
        + f"-asvd{int(args.param_ratio_target*100)}"
    )
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    config = model.config.to_dict()
    config["truncation_ranks"] = {}
    for name, module in model.named_modules():
        if isinstance(module, SVDLinear):
            config["truncation_ranks"][name] = module.truncation_rank
    if "opt" in model_id:
        config["auto_map"] = {
            "AutoConfig": "configuration_asvd_opt.ASVDOPTConfig",
            "AutoModelForCausalLM": "modeling_asvd_opt.ASVDOPTForCausalLM",
        }
        config["architectures"] = ["ASVDOPTForCausalLM"]
        os.system(
            "cp ./huggingface_repos/configuration_asvd_opt.py ./huggingface_repos/modeling_asvd_opt.py ./"
            + save_path
        )
    elif "llama" in model_id:
        config["auto_map"] = {
            "AutoConfig": "configuration_asvd_llama.ASVDLlamaConfig",
            "AutoModelForCausalLM": "modeling_asvd_llama.ASVDLlamaForCausalLM",
        }
        config["architectures"] = ["ASVDLlamaForCausalLM"]
        os.system(
            "cp ./huggingface_repos/configuration_asvd_llama.py ./huggingface_repos/modeling_asvd_llama.py ./"
            + save_path
        )
    import json

    json.dump(config, open(save_path + "/config.json", "w"), indent=2)

    print("Done building huggingface model")
    del model
    del tokenizer
    if args.push:
        # load
        hub_name = model_id.split("/")[-1] + f"-asvd{int(args.param_ratio_target*100)}"
        tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            save_path,
            device_map="cpu",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer.push_to_hub(hub_name)
        model.push_to_hub(hub_name)


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
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
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
        choices=["none", "rtn_int8", "rtn_int6"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="push to hub",
    )
    args = parser.parse_args()

    main(args)
