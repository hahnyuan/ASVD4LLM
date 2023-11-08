import os
import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def calib_input_distribution(model, calib_loader):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_input_distribution.pt"
    if os.path.exists(cache_file):
        all_input_abs_mean = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.input_abs_mean = all_input_abs_mean[name].to(
                    module.weight.device
                )
        return
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
        module.input_abs_mean += abs_mean
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.input_abs_mean += abs_max

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.input_abs_mean = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save input_abs_mean
    all_input_abs_mean = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_input_abs_mean[name] = module.input_abs_mean
    torch.save(all_input_abs_mean, cache_file)


@torch.no_grad()
def calib_input_output_distribution(model, calib_loader):
    model_id = model.config._name_or_path
    input_cache_file = f"cache/{model_id.replace('/','_')}_calib_input_distribution2.pt"
    output_cache_file = (
        f"cache/{model_id.replace('/','_')}_calib_output_distribution2.pt"
    )
    if os.path.exists(input_cache_file) and os.path.exists(output_cache_file):
        all_input_abs_mean, all_input_std, all_input_mean = torch.load(
            input_cache_file, map_location="cpu"
        )

        all_output_abs_mean, all_output_std, all_output_mean = torch.load(
            output_cache_file, map_location="cpu"
        )
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.input_abs_mean = all_input_abs_mean[name].to(
                    module.weight.device
                )
                module.input_std = all_input_std[name].to(module.weight.device)
                module.input_mean = all_input_mean[name].to(module.weight.device)
                module.output_abs_mean = all_output_abs_mean[name].to(
                    module.weight.device
                )
                module.output_std = all_output_std[name].to(module.weight.device)
                module.output_mean = all_output_mean[name].to(module.weight.device)
        return
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        abs_sum = input[0].abs().sum(dim=-2).detach().view(-1)
        module.input_abs_sum += abs_sum
        input_sum = input[0].sum(dim=-2).detach().view(-1)
        module.input_sum += input_sum
        input_pow2_sum = input[0].pow(2).sum(dim=-2)
        module.input_pow2_sum += input_pow2_sum.detach().view(-1)

        abs_sum = output.abs().sum(dim=-2).detach().view(-1)
        module.output_abs_sum += abs_sum
        output_sum = output.sum(dim=-2).detach().view(-1)
        module.output_sum += output_sum
        output_pow2_sum = output.pow(2).sum(dim=-2)
        module.output_pow2_sum += output_pow2_sum.detach().view(-1)

        module.cnt+=output.size(0)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.cnt=0
            module.input_abs_sum = 0
            module.input_sum = 0
            module.input_pow2_sum = 0
            module.output_abs_sum = 0
            module.output_sum = 0
            module.output_pow2_sum = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save input_abs_mean
    all_input_abs_mean = {}
    all_input_std = {}
    all_input_mean = {}
    all_output_abs_mean = {}
    all_output_std = {}
    all_output_mean = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            module.input_abs_mean = module.input_abs_sum / module.cnt
            module.input_mean = module.input_sum / module.cnt
            module.input_std = torch.sqrt(
                module.input_pow2_sum / module.cnt - module.input_mean.pow(2)
            )
            module.output_abs_mean = module.output_abs_sum / module.cnt
            module.output_mean = module.output_sum / module.cnt
            module.output_std = torch.sqrt(
                module.output_pow2_sum / module.cnt - module.output_mean.pow(2)
            )
            all_input_abs_mean[name] = module.input_abs_mean
            all_input_std[name] = module.input_std
            all_input_mean[name] = module.input_mean
            all_output_abs_mean[name] = module.output_abs_mean
            all_output_std[name] = module.output_std
            all_output_mean[name] = module.output_mean

    torch.save((all_input_abs_mean, all_input_std, all_input_mean), input_cache_file)
    torch.save(
        (all_output_abs_mean, all_output_std, all_output_mean), output_cache_file
    )
