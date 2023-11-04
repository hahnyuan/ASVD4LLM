# Run
CUDA_VISIBLE_DEVICES='0,1' python svd_lora_train.py --model_id="huggyllama/llama-7b" --rank_compress_ratio=0.2 --lora_method "Uonly"
CUDA_VISIBLE_DEVICES='2,3' python svd_lora_train.py --model_id="huggyllama/llama-7b" --rank_compress_ratio=0.2 --lora_method "Vonly"
CUDA_VISIBLE_DEVICES='2' python svd_lora_train.py --model_id="facebook/opt-125m" --lora_method "Uonly" --rank_compress_ratio=0.2
CUDA_VISIBLE_DEVICES='0' python svd_lora_train.py --model_id="facebook/opt-1.3b" --lora_method "Uonly" --rank_compress_ratio=0.2
CUDA_VISIBLE_DEVICES='1' python svd_lora_train.py --model_id="facebook/opt-1.3b" --lora_method "Vonly" --rank_compress_ratio=0.2
CUDA_VISIBLE_DEVICES='2' python svd_lora_train.py --model_id="facebook/opt-1.3b" --lora_method "UV" --rank_compress_ratio=0.2

CUDA_VISIBLE_DEVICES='0' python act_aware_svd_lora_train.py --model_id="facebook/opt-1.3b" --lora_method "Uonly" --rank_compress_ratio=0.2
CUDA_VISIBLE_DEVICES='1' python act_aware_svd_lora_train.py --model_id="facebook/opt-1.3b" --lora_method "Vonly" --rank_compress_ratio=0.2
CUDA_VISIBLE_DEVICES='2' python act_aware_svd_lora_train.py --model_id="facebook/opt-1.3b" --lora_method "UV" --rank_compress_ratio=0.2
CUDA_VISIBLE_DEVICES='3' python act_aware_svd_lora_train.py --model_id="facebook/opt-1.3b" --lora_method "UVall" --rank_compress_ratio=0.2

CUDA_VISIBLE_DEVICES='2' python act_aware_svd_lora_train.py --model_id="facebook/opt-125m" --lora_method "Uonly" --rank_compress_ratio=0.2

CUDA_VISIBLE_DEVICES='2' python svd_lora_train.py --model_id="facebook/opt-1.3b" --lora_method "reconstruct"

# Eval
CUDA_VISIBLE_DEVICES='0' python tools/eval_checkpoint.py "huggyllama/llama-7b" output/svd_lora_train/checkpoint-2000 --rank_compress_ratio=0.2 --lora_method "UV"
CUDA_VISIBLE_DEVICES='0' python tools/eval_checkpoint.py "huggyllama/llama-7b" output/svd_lora_train_Uonly_0.2/checkpoint-2000 --rank_compress_ratio=0.2 --lor`a_method "Uonly" --limit 200
CUDA_VISIBLE_DEVICES='1' python tools/eval_checkpoint.py --model_id="huggyllama/llama-7b" --path=output/svd_lora_train_Vonly_0.2/checkpoint-2000 --rank_compress_ratio=0.2 --lora_method "Vonly"

CUDA_VISIBLE_DEVICES='2' python tools/eval_checkpoint.py --model_id="facebook/opt-1.3b" --lora_method "UV" --rank_compress_ratio=0.2 --path="output/facebook_opt-1.3b_svd_lora_train_UV_0.2/checkpoint-3000"
CUDA_VISIBLE_DEVICES='1' python tools/eval_checkpoint.py --model_id="facebook/opt-1.3b" --lora_method "Uonly" --rank_compress_ratio=0.2 --path="output/facebook_opt-1.3b_svd_lora_train_Uonly_0.2/checkpoint-3000"
CUDA_VISIBLE_DEVICES='0' python tools/eval_checkpoint.py --model_id="facebook/opt-1.3b" --lora_method "Vonly" --rank_compress_ratio=0.2 --path="output/facebook_opt-1.3b_svd_lora_train_Vonly_0.2/checkpoint-3000"


CUDA_VISIBLE_DEVICES='0' python tools/eval_checkpoint.py --model_id="facebook/opt-1.3b" --lora_method "Uonly" --rank_compress_ratio=0.2 --path="output/facebook_opt-1.3b_act_aware_Uonly_0.2/checkpoint-3000"

output/facebook_opt-1.3b_svd_lora_train_reconstruct_0.3_0.1/checkpoint-3000

CUDA_VISIBLE_DEVICES='2' python tools/eval_checkpoint.py --model_id="facebook/opt-1.3b" --lora_method "reconstruct" --path="tmp/LLM_PEFT/svd_peft/checkpoints/llama-7b-svdlora/final_merged"

# Sensitivity
CUDA_VISIBLE_DEVICES='0' python experiments/sensitivity.py --model_id="facebook/opt-125m" --lora_method "reconstruct"

CUDA_VISIBLE_DEVICES='1' python experiments/sensitivity.py --model_id="facebook/opt-1.3b" --lora_method "reconstruct"

CUDA_VISIBLE_DEVICES='3' python experiments/sensitivity.py --model_id="facebook/opt-1.3b" --lora_method "reconstruct" --act_aware

CUDA_VISIBLE_DEVICES='2' python experiments/sensitivity.py --model_id="huggyllama/llama-7b" --lora_method "reconstruct"

CUDA_VISIBLE_DEVICES='3' python experiments/sensitivity.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --act_aware

CUDA_VISIBLE_DEVICES='3' python experiments/sensitivity.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --act_aware

# Mixed rank
CUDA_VISIBLE_DEVICES='3' python mixed_rank_svd_lora_train.py --model_id="facebook/opt-125m" --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --ppl_thresh 29 --act_aware --lora_method "reconstruct"
