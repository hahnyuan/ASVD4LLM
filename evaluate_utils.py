import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
import os

from datautils import get_eval_loaders
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


@torch.no_grad()
def evaluate_perplexity(model, dataset, limit):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []

    for i in range(nsamples):
        if i == limit:
            break
        input_ids = dataset[i : i + 1, :-1].to(model.device)
        labels = dataset[i : i + 1, 1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return ppl.item()


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    model_name,
    tasks,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    ppl_seqlen=2048,
):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str tasks are split by ,
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    """
    # lm = EvalLM(model, tokenizer, batch_size=batch_size)
    results = {}
    if eval_ppl:
        for dataset in eval_ppl.split(","):
            cache_testloader = f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                # print(f"load calibration from {cache_testloader}")
            else:
                testloader = get_eval_loaders(dataset, tokenizer)
                torch.save(testloader, cache_testloader)
            # print(dataset)
            testenc = testloader.input_ids
            nsamples = testenc.numel() // ppl_seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.eval()
            nlls = []

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * ppl_seqlen) : ((i + 1) * ppl_seqlen)].to(model.device)
                outputs = model.model(batch)
                hidden_states = outputs[0]  # .to(lm.model.lm_head.weight.device)
                logits = model.lm_head(hidden_states)  # .contiguous()
                shift_logits = logits[:, :-1, :]  # .contiguous()
                shift_labels = testenc[:, (i * ppl_seqlen) : ((i + 1) * ppl_seqlen)][:, 1:].to(model.device)
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0).mean()
                neg_log_likelihood = loss.float() * ppl_seqlen
                nlls.append(neg_log_likelihood)
                if i == limit:
                    break
                # if i == 1:
                #     print(
                #         "memory_allocated",
                #         i,
                #         torch.cuda.memory_allocated() / 1024 / 1024,
                #         "max memory_allocated",
                #         torch.cuda.max_memory_allocated() / 1024**2,
                #     )

            ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * ppl_seqlen))
            print(dataset, ppl.item())
            model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    if tasks == "longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=full_longeval_datasets)
        results.update(longbench_results)
        tasks = ""
    elif tasks == "small_longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=small_longeval_datasets)
        results.update(longbench_results)
        tasks = ""
    if tasks == "mmlu":
        t_results = evaluator.simple_evaluate(
            HFLM(pretrained=model, tokenizer=tokenizer),
            tasks=tasks.split(","),
            batch_size="auto",
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
        )
        results["mmlu"] = t_results["results"]["mmlu"]
    elif tasks != "":
        t_results = evaluator.simple_evaluate(
            HFLM(pretrained=model, tokenizer=tokenizer),
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        t_results = t_results["results"]
        acc_list = [t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]]
        t_results["mean"] = sum(acc_list) / len(acc_list)
        results.update(t_results)
        print(results)
        # print mean
        print(f"\n\n===== mean acc: {sum(acc_list)/len(acc_list)} =====\n\n")

    return results
