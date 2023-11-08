import os
import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def calib_input_distribution(model, calib_loader):
    model_id=model.config._name_or_path
    cache_file=f"cache/{model_id.replace('/','_')}_calib_input_distribution.pt"
    if os.path.exists(cache_file):
        all_input_abs_mean=torch.load(cache_file,map_location='cpu')
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.input_abs_mean = all_input_abs_mean[name].to(module.weight.device)
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
    all_input_abs_mean={}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_input_abs_mean[name]=module.input_abs_mean
    torch.save(all_input_abs_mean,cache_file)

@torch.no_grad()
def calib_input_output_distribution(model, calib_loader):
    model_id=model.config._name_or_path
    cache_file=f"cache/{model_id.replace('/','_')}_calib_input_distribution.pt"
    output_cache_file=f"cache/{model_id.replace('/','_')}_calib_output_distribution.pt"
    if os.path.exists(cache_file) and os.path.exists(output_cache_file):
        all_input_abs_mean=torch.load(cache_file,map_location='cpu')
        all_output_abs_mean=torch.load(output_cache_file,map_location='cpu')
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.input_abs_mean = all_input_abs_mean[name].to(module.weight.device)
                module.output_abs_mean = all_output_abs_mean[name].to(module.weight.device)
        return
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
        module.input_abs_mean += abs_mean
        abs_mean = output.abs().mean(dim=-2).detach().view(-1)
        module.output_abs_mean += abs_mean

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.input_abs_mean = 0
            module.output_abs_mean = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save input_abs_mean
    all_input_abs_mean={}
    all_output_abs_mean={}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_input_abs_mean[name]=module.input_abs_mean
            all_output_abs_mean[name]=module.output_abs_mean
    
    torch.save(all_input_abs_mean,cache_file)
    torch.save(all_output_abs_mean,output_cache_file)