# ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models

This work explores a novel paradigm for reducing the memory footprint of LLMs to facilitate their wider adoption in various computing environments. We delve into the challenges of traditional low-rank decomposition methods in LLM compression, notably their dependency on extensive training data and computational resources. Addressing these limitations, we propose a training-free approach, including an innovative technique, Activation-aware Singular Value Decomposition (ASVD). ASVD effectively manages weight matrix outliers by adjusting values based on the activation distribution, improving decomposition accuracy and efficiency. Our method also addresses the varying sensitivity of different LLM layers to decomposition, with an iterative calibration process for optimal layer-specific decomposition. Experiments demonstrate that ASVD can compress network by 10\%-20\% without losing reasoning capacities. Additionally, it seamlessly integrates with quantization, showcasing its compatibility.

# requirement
- python>=3.10
- pytorch>=2.1.0
- transformers>=4.35.0

# direct use

Some of the decomposed models are uploaded to huggingface hub. You can download and load them using the following code:

```python3
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

# usage

You can use the following command to run the ASVD. This will take several hours to generate the sensitivity of each layer. The sensitivity will be saved in the cache file. 
The time will be reduced to several minutes if you use the cache file.

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
```

You can use the cache file to omit the calibration process. The cache file can be downloaded from huggingface-hub, by using the following command:
```
git clone https://huggingface.co/hahnyuan/ASVD4LLM_sensitivity_cache cache
```
Or download the cache file from [here](https://huggingface.co/hahnyuan/ASVD4LLM_sensitivity_cache/tree/main) yourself. And place the cache file in the `cache` folder.