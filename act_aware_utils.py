import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


def calib_fisher_info(model, calib_loader, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"
    if os.path.exists(cache_file) and use_cache:
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info += module.weight.grad.detach().pow(2).mean(0)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info
    torch.save(all_fisher_info, cache_file)


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method):
    model.eval()

    # set hook for every Linear layers
    def hook(module, input, output):
        if "magnitude" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.magnitude += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.magnitude = torch.where(
                abs_max > module.magnitude,
                abs_max,
                module.magnitude,
            )
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.magnitude += abs_max

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.magnitude = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save magnitude
    all_magnitude = []
    alpha = 0.5 if "alpha0.5" in method else 1
    if "opt" in model.config._name_or_path:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    for i, layer in enumerate(layers):
        layer_magnitude = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                module._forward_hooks.clear()
                layer_magnitude[f"{name}"] = module.magnitude**alpha
                module.magnitude = None
        all_magnitude.append(layer_magnitude)
    return all_magnitude


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


@torch.no_grad()
def layerwise_cholesky_decomposition(model_name, model, calib_loader, dev):
    # modified from SVD-LLM
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if kwargs["attention_mask"] is None:
                kwargs["attention_mask"] = torch.ones_like(kwargs["position_ids"])
            else:
                print(kwargs["attention_mask"])
            inps[cache["i"]] = inp.cpu()
            cache["i"] += 1
            if cache["attention_mask"] is None:
                cache["attention_mask"] = kwargs["attention_mask"].cpu()
                if "opt" not in model_name:
                    cache["position_ids"] = kwargs["position_ids"].cpu()
            else:
                cache["attention_mask"] = torch.cat((cache["attention_mask"], kwargs["attention_mask"].cpu()), dim=0)
                if "opt" not in model_name:
                    cache["position_ids"] = torch.cat((cache["position_ids"], kwargs["position_ids"].cpu()), dim=0)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache["attention_mask"]
    if "opt" not in model_name:
        position_ids = cache["position_ids"]
    all_cholesky_mat = {}
    for i in tqdm(range(len(layers))):
        layer_cholesky_mat = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1, 2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.XmulXT += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()

        handles = []
        for name in subset:
            subset[name].XmulXT = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if "opt" in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
            elif "Qwen" in model_name:
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    position_ids=position_ids[j].unsqueeze(0).to(dev),
                )[0]
            else:
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    # attention_mask=attention_masks[j].unsqueeze(0).to(dev),
                    position_ids=position_ids[j].unsqueeze(0).to(dev),
                )[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].XmulXT = subset[name].XmulXT.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            XmulXT = subset[name].XmulXT.double().to(dev)
            try:
                cholesky_mat = torch.linalg.cholesky(XmulXT)
            except Exception as e:
                print("Warning: eigen XmulXT is not positive!")
                eigenvalues = torch.linalg.eigvalsh(XmulXT)
                XmulXT += (-eigenvalues[0] + 1e-6) * torch.eye(XmulXT.shape[0]).to(dev)
                cholesky_mat = torch.linalg.cholesky(XmulXT)
                eigenvalues = None
                del eigenvalues
            layer_cholesky_mat[name] = cholesky_mat.cpu()
            cholesky_mat = XmulXT = subset[name].XmulXT = None
            del cholesky_mat, XmulXT, subset[name].XmulXT
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        all_cholesky_mat[i] = layer_cholesky_mat
        inps = outs
        torch.cuda.empty_cache()
    return all_cholesky_mat
