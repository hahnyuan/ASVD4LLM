import argparse
import os
import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from evaluate import evaluate_model
from modules.svd_linear import SVDLinear

from utils import print_gpu_memory
from datautils import get_calib_data, sample_train_loaders
from svd_init_utils import calib_input_distribution, calib_input_output_distribution
from sensitivity import calib_sensitivity
from quantization import Quantizer, GPTQ

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

def run_eval(model, tokenizer, sensitivity_dict, args, log_file, act_aware=False):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    for target_params_ratio in [0.95, 0.9, 0.85, 0.8, 0.75]:
        high = len(sorted_sensitive_list) - 1
        low = 0
        while low < high:
            mid = (low + high) // 2
            layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
            for layername, ratio, ppl in sorted_sensitive_list[mid:]:
                layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
            tot_params = 0
            compress_params = 0
            for layername, ratio in layers_min_ratio.items():
                # set ratio
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            param_ratio = compress_params / tot_params
            if param_ratio > target_params_ratio:
                high = mid
            else:
                param_ratio = mid + 1
        for layername, ratio in layers_min_ratio.items():
            # set ratio
            raw_linear = module_dict[layername]
            info = linear_info[raw_linear]
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=ratio,
                alpha=args.alpha,
                act_aware=act_aware,
                oc_split=args.test_split
                if raw_linear.in_features < raw_linear.out_features
                else 1,
                ic_split=args.test_split
                if raw_linear.in_features > raw_linear.out_features
                else 1,
            )
            setattr(info["father"], info["name"], svd_linear)
        result = evaluate_model(
            model,
            tokenizer,
            args.model_id,
            "",
            eval_ppl="wikitext2,ptb",
            limit=-1,
        )
        msg=f"act-aware{act_aware} target_params_ratio={target_params_ratio}\n"
        print(msg)
        print(result)
        log_file.write(msg)
        log_file.write(str(result))
        log_file.flush()


def main(args):
    model_id = args.model_id

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )

    model = model.to_bettertransformer()

    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "",
        eval_ppl="wikitext2,ptb",
        limit=-1,
    )
    print(result)
    save_path = f"output/final/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = open(f"{save_path}/{args.model_id.replace('/','_')}.json", "a+")
    log_file.write("original\n")
    log_file.write(str(result))
    log_file.flush()

    cablib_dataset = "wikitext2"
    calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, 256)
    calib_input_distribution(model, calib_loader, args.scaling_method)
    sensitivity = calib_sensitivity(model, tokenizer, args)
    # calib_input_output_distribution(model, calib_loader)
    # train_input_output_scale(model, calib_loader)
    # calib_full_input(model, calib_loader)
    print_gpu_memory("before convert_to_svd_linear")
    run_eval(model, tokenizer, sensitivity, args, log_file, act_aware=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--test_split",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max"],
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
