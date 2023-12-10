# ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models

This work explores a novel paradigm for reducing the memory footprint of LLMs to facilitate their wider adoption in various computing environments. We delve into the challenges of traditional low-rank decomposition methods in LLM compression, notably their dependency on extensive training data and computational resources. Addressing these limitations, we propose a training-free approach, including an innovative technique, Activation-aware Singular Value Decomposition (ASVD). ASVD effectively manages weight matrix outliers by adjusting values based on the activation distribution, improving decomposition accuracy and efficiency. Our method also addresses the varying sensitivity of different LLM layers to decomposition, with an iterative calibration process for optimal layer-specific decomposition. Experiments demonstrate that ASVD can compress network by 10\%-20\% without losing reasoning capacities. Additionally, it seamlessly integrates with quantization, showcasing its compatibility.

For more details, please read our paper.

# Requirement
- python>=3.10
- pip install -r requirements.txt

# Direct usage

Some of the decomposed models are uploaded to huggingface hub. You can directly download and use them using the following code:

```python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "hahnyuan/opt-125m-asvd90"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
```

Now supported models (asvd90 means target param ratio=90%):
- hahnyuan/opt-125m-asvd90 
- hahnyuan/Llama-2-7b-hf-asvd95
- hahnyuan/Llama-2-7b-hf-asvd90
- hahnyuan/Llama-2-7b-hf-asvd85
- hahnyuan/Llama-2-13b-hf-asvd95
- hahnyuan/Llama-2-13b-hf-asvd90
- hahnyuan/Llama-2-13b-hf-asvd85

You can quantize these models using the tools that transformers provided, for example:
```python3
# 4bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

# 8bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    load_in_8bit=True,
)
```

# Run ASVD

You can use the following command to run the ASVD. This will take several hours to generate the sensitivity of each layer. The sensitivity will be saved in the cache file. 
The time will be reduced to several minutes if you use the cache file.

NOTE: A dedicated calibration dataset is necessary for chat models like Llama-2-7b-chat-hf. Failure to create such a dataset may lead to suboptimal performance. We currently do not provide a calibration dataset for chat models. You can write your own code in the `get_calib_data` function of the `datautils.py` file to generate the calibration dataset for chat models.

```
usage: asvd.py [-h] [--model_id MODEL_ID] [--ppl_target PPL_TARGET] [--param_ratio_target PARAM_RATIO_TARGET] [--act_aware] [--alpha ALPHA] [--n_calib_samples N_CALIB_SAMPLES] [--calib_dataset {wikitext2,c4,ptb}]
               [--scaling_method {abs_mean,abs_max,fisher}] [--use_cache]

options:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   Pretrained model ID
  --ppl_target PPL_TARGET
                        target ppl
  --param_ratio_target PARAM_RATIO_TARGET
                        target param ratio
  --act_aware           use act aware svd (ASVD)
  --alpha ALPHA         hyper-parameter alpha for ASVD
  --n_calib_samples N_CALIB_SAMPLES
                        number of samples used for calibration
  --calib_dataset {wikitext2,c4,ptb}
                        calibration dataset
  --scaling_method {abs_mean,abs_max,fisher}
                        scaling method
  --use_cache           use cached calibration results
  --weight_quant {none,rtn_int8,rtn_int6}
                        weight quantization method
```


Examples:
```
CUDA_VISIBLE_DEVICES='0' python asvd.py --model_id="facebook/opt-125m" --act_aware --alpha 0.5 --n_calib_samples 16 --scaling_method abs_mean --ppl_target 40 --use_cache

CUDA_VISIBLE_DEVICES='1' python asvd.py --model_id="facebook/opt-125m" --act_aware --alpha 1 --n_calib_samples 16 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache

CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="meta-llama/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache

CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="meta-llama/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --eval_mmlu

CUDA_VISIBLE_DEVICES='0' python asvd.py --model_id="meta-llama/Llama-2-7b-chat-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache

CUDA_VISIBLE_DEVICES='1' python asvd.py --model_id="meta-llama/Llama-2-13b-chat-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache

```

You can use the cache file to omit the calibration process. The cache file can be downloaded from huggingface-hub, by using the following command:
```
git clone https://huggingface.co/hahnyuan/ASVD4LLM_sensitivity_cache cache
```
Or download the cache file from [here](https://huggingface.co/hahnyuan/ASVD4LLM_sensitivity_cache/tree/main) yourself. And place the cache file in the `cache` folder.

## Making huggingface repository

You can use the following command to make a huggingface repository for your ASVD model. 

```
usage: huggingface_repos/build_asvd_repo.py [-h] [--model_id MODEL_ID] [--ppl_target PPL_TARGET] [--param_ratio_target PARAM_RATIO_TARGET] [--act_aware]
                          [--alpha ALPHA] [--n_calib_samples N_CALIB_SAMPLES] [--calib_dataset {wikitext2,c4,ptb}]
                          [--scaling_method {abs_mean,abs_max,fisher,fisher_abs_mean}] [--sensitivity_metric {ppl,stable_rank}] [--use_cache]
                          [--weight_quant {none,rtn_int8,rtn_int6}] [--eval_mmlu] [--sigma_fuse {U,V,UV}] [--push]

options:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   Pretrained model ID
  --ppl_target PPL_TARGET
                        target ppl
  --param_ratio_target PARAM_RATIO_TARGET
                        target param ratio
  --act_aware           use act aware svd (ASVD)
  --alpha ALPHA         hyper-parameter alpha for ASVD
  --n_calib_samples N_CALIB_SAMPLES
                        number of samples used for calibration
  --calib_dataset {wikitext2,c4,ptb}
                        calibration dataset
  --scaling_method {abs_mean,abs_max,fisher,fisher_abs_mean}
                        scaling method
  --sensitivity_metric {ppl,stable_rank}
                        search metric
  --use_cache           use cached calibration results
  --weight_quant {none,rtn_int8,rtn_int6}
                        weight quantization method
  --eval_mmlu           evaluate mmlu
  --sigma_fuse {U,V,UV}
                        sigma fuse method
  --push                push to hub
```

Examples:
```
CUDA_VISIBLE_DEVICES='0' python huggingface_repos/build_asvd_repo.py --model_id="facebook/opt-125m" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
```

This will generate a huggingface repository in the `huggingface_repos` folder. You can use this repository directly:
```python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "huggingface_repos/opt-125m-asvd90"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
```

# Citation

Please cite our paper if you use ASVD.