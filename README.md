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

CUDA_VISIBLE_DEVICES='2' python tools/eval_checkpoint.py --model_id="facebook/opt-1.3b" --lora_method "reconstruct" --path="output/facebook_opt-1.3b_mixed_rank_UV_14.8_True"


CUDA_VISIBLE_DEVICES='3' python tools/eval_checkpoint.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --path="output/facebook_opt-125m_mixed_rank_reconstruct_29.0_True/final_merged"



# Sensitivity
CUDA_VISIBLE_DEVICES='0' python experiments/sensitivity.py --model_id="facebook/opt-125m" --lora_method "reconstruct"

CUDA_VISIBLE_DEVICES='2' python experiments/sensitivity.py --model_id="facebook/opt-1.3b" --lora_method "reconstruct"

CUDA_VISIBLE_DEVICES='3' python experiments/sensitivity.py --model_id="facebook/opt-1.3b" --lora_method "reconstruct" --act_aware

CUDA_VISIBLE_DEVICES='2' python experiments/sensitivity.py --model_id="huggyllama/llama-7b" --lora_method "reconstruct"

CUDA_VISIBLE_DEVICES='3' python experiments/sensitivity.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --act_aware

CUDA_VISIBLE_DEVICES='3' python experiments/sensitivity.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --act_aware

CUDA_VISIBLE_DEVICES='0' python experiments/sensitivity.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --act_aware

# Mixed rank
CUDA_VISIBLE_DEVICES='3' python mixed_rank_svd_lora_train.py --model_id="facebook/opt-125m" --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --ppl_thresh 29 --act_aware --lora_method "reconstruct"

CUDA_VISIBLE_DEVICES='2' python mixed_rank_svd_lora_train.py --model_id="facebook/opt-1.3b" --sensitivity_json output/sensitivity_facebook_opt-1.3b_False.json --ppl_thresh 14.8 --act_aware --lora_method "reconstruct"

CUDA_VISIBLE_DEVICES='0' python mixed_rank_svd_lora_train.py --model_id="facebook/opt-125m" --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --ppl_thresh 29 --act_aware --lora_method "UV"

CUDA_VISIBLE_DEVICES='3' python mixed_rank_svd_lora_train.py --model_id="facebook/opt-1.3b" --sensitivity_json output/sensitivity_facebook_opt-1.3b_False.json --ppl_thresh 14.8 --act_aware --lora_method "UV"

# single block sensitivity
CUDA_VISIBLE_DEVICES='0' python experiments/single_block_sensitivity.py --model_id="facebook/opt-125m" --layer_name model.decoder.layers.0. --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --lora_method "reconstruct" --ppl_thresh 28 --act_aware

CUDA_VISIBLE_DEVICES='1' python experiments/single_block_sensitivity.py --model_id="facebook/opt-125m" --layer_name model.decoder.layers.0. --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --lora_method "reconstruct" --ppl_thresh 28.5 --act_aware

CUDA_VISIBLE_DEVICES='2' python experiments/single_block_sensitivity.py --model_id="facebook/opt-125m" --layer_name model.decoder.layers.0. --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --lora_method "reconstruct" --ppl_thresh 29 --act_aware

CUDA_VISIBLE_DEVICES='0' python experiments/single_block_sensitivity.py --model_id="facebook/opt-125m" --layer_name v_proj --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --lora_method "reconstruct" --ppl_thresh 28 --act_aware

CUDA_VISIBLE_DEVICES='1' python experiments/single_block_sensitivity.py --model_id="facebook/opt-125m" --layer_name v_proj --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --lora_method "reconstruct" --ppl_thresh 28.5 --act_aware

CUDA_VISIBLE_DEVICES='2' python experiments/single_block_sensitivity.py --model_id="facebook/opt-125m" --layer_name v_proj --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --lora_method "reconstruct" --ppl_thresh 29 --act_aware

# sequential svd rank search


CUDA_VISIBLE_DEVICES='0' python experiments/sequential_svd_rank_search.py --model_id="facebook/opt-125m" --layer_name model.decoder.layers.5. --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --lora_method "reconstruct" --act_aware

CUDA_VISIBLE_DEVICES='3' python experiments/sequential_svd_rank_search.py --model_id="facebook/opt-125m" --layer_name model.decoder.layers.9. --sensitivity_json output/sensitivity_facebook_opt-125m_True.json --lora_method "reconstruct" --act_aware

# Greedy
CUDA_VISIBLE_DEVICES='0' python experiments/greedy.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --act_aware --ppl_target_st 30 --ppl_target_ed 40

CUDA_VISIBLE_DEVICES='2' python experiments/greedy.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --act_aware --ppl_target_st 30 --ppl_target_ed 35

CUDA_VISIBLE_DEVICES='1' python experiments/greedy.py --model_id="facebook/opt-1.3b" --lora_method "UV" --act_aware --ppl_target_st 15.8 --ppl_target_ed 20

CUDA_VISIBLE_DEVICES='3' python experiments/greedy.py --model_id="huggyllama/llama-7b" --lora_method "reconstruct" --act_aware --ppl_target_st 5.8 --ppl_target_ed 8

CUDA_VISIBLE_DEVICES='2' python experiments/greedy.py --model_id="huggyllama/llama-7b" --lora_method "reconstruct" --act_aware --ppl_target_st 5.8 --ppl_target_ed 6.5

## Greedy Train
CUDA_VISIBLE_DEVICES='0' python experiments/greedy_train.py --model_id="facebook/opt-125m" --lora_method "UV" --act_aware --ppl_target_st 30 --ppl_target_ed 40


# Eval after greedy

CUDA_VISIBLE_DEVICES='2' python experiments/test_after_greedy.py --model_id="facebook/opt-125m" --act_aware --ratios "[1, 0.8125, 0.6875, 1, 0.5625, 1, 1, 0.625, 0.3125, 0.9375, 0.4375, 0.75, 0.875, 1, 0.9375, 1, 0.625, 1, 0.9375, 0.875, 0.625, 1, 0.8125, 0.75, 1, 1, 0.4375, 1, 0.625, 0.625, 1, 0.75, 0.625, 1, 0.5, 0.9375, 1, 0.875, 0.8125, 1, 0.5625, 1, 1, 0.9375, 0.75, 0.9375, 0.75, 1, 1, 0.75, 0.9375, 0.9375, 0.625, 1, 1, 1, 0.75, 0.75, 0.625, 0.625, 1, 1, 0.375, 0.625, 0.5625, 0.625, 1, 1, 1, 1, 1, 1]"

CUDA_VISIBLE_DEVICES='2' python experiments/viz_after_greedy.py --model_id="facebook/opt-1.3b" --lora_method "UV" --act_aware --ratios "[1, 1, 0.4, 0.4, 0.8, 0.4, 0.4, 1, 1, 0.6, 1, 0.4, 0.4, 1, 0.8, 0.8, 0.8, 1, 0.6, 0.8, 0.6, 1, 0.8, 0.4, 0.8, 0.6, 0.4, 0.4, 1, 0.2, 0.4, 0.4, 0.4, 0.8, 1, 0.6, 0.6, 0.8, 1, 0.2, 0.8, 0.4, 0.6, 1, 1, 0.6, 0.6, 0.8, 1, 0.6, 1, 0.8, 0.8, 0.6, 0.8, 1, 0.8, 0.6, 1, 1, 1, 1, 0.6, 0.2, 0.8, 0.4, 1, 1, 0.6, 1, 0.6, 1, 0.6, 0.8, 1, 0.8, 0.8, 0.6, 0.4, 1, 0.8, 0.4, 0.8, 0.8, 1, 1, 0.8, 0.2, 0.8, 0.1, 1, 1, 1, 0.4, 1, 0.4, 0.4, 1, 0.8, 0.6, 0.6, 0.4, 0.8, 1, 0.8, 0.4, 0.6, 0.4, 0.6, 0.8, 0.6, 0.8, 0.4, 1, 0.1, 1, 0.8, 0.4, 0.4, 0.4, 0.6, 0.8, 0.8, 0.4, 0.2, 0.4, 0.2, 0.6, 1, 0.2, 0.1, 0.4, 0.1, 1, 0.8, 0.1, 0.2, 0.1, 0.2, 1, 1, 0.1, 0.4, 0.2, 0.4]"
CUDA_VISIBLE_DEVICES='1' python experiments/viz_after_greedy.py --model_id="facebook/opt-1.3b" --lora_method "UV" --act_aware --ratios "[1, 1, 0.8, 0.4, 1, 0.4, 0.4, 1, 1, 0.8, 1, 0.8, 0.8, 1, 0.8, 1, 0.8, 1, 1, 0.8, 0.6, 1, 0.8, 0.6, 1, 0.8, 0.6, 0.4, 1, 0.6, 0.6, 0.2, 1, 1, 1, 0.4, 0.8, 1, 1, 0.4, 1, 0.2, 0.8, 1, 1, 0.6, 1, 0.8, 1, 0.6, 1, 0.6, 0.8, 0.6, 0.8, 1, 1, 0.6, 1, 0.6, 1, 1, 0.6, 0.8, 1, 0.2, 0.6, 1, 0.8, 0.6, 0.8, 0.6, 0.6, 1, 1, 0.6, 0.8, 1, 0.8, 1, 0.8, 0.8, 0.8, 0.6, 1, 1, 0.8, 0.6, 0.8, 0.6, 0.4, 1, 1, 0.4, 1, 0.4, 1, 1, 0.8, 0.6, 1, 0.6, 0.4, 0.6, 0.8, 0.4, 1, 0.4, 0.4, 0.8, 0.6, 0.4, 0.1, 0.1, 0.1, 0.6, 0.8, 0.6, 0.4, 0.8, 0.1, 0.8, 1, 0.4, 0.6, 0.4, 0.2, 0.6, 1, 0.2, 0.2, 0.4, 0.2, 0.8, 0.8, 0.1, 0.4, 0.4, 0.2, 1, 1, 0.1, 0.8, 0.4, 0.4]"

CUDA_VISIBLE_DEVICES='1' python experiments/viz_after_greedy.py --model_id="huggyllama/llama-7b" --lora_method "UV" --act_aware --ratios "[1, 0.5625, 1, 0.75, 0.375, 0.3125, 0.375, 0.6875, 1, 1, 1, 0.1875, 0.25, 1, 0.375, 1, 1, 0.9375, 0.25, 0.375, 1, 0.6875, 1, 1, 0.875, 1, 0.8125, 1, 0.875, 1, 1, 0.8125, 0.375, 1, 1, 0.625, 1, 1, 0.8125, 0.9375, 0.8125, 0.5, 1, 0.875, 1, 1, 0.4375, 0.5, 1, 0.3125, 1, 1, 0.9375, 0.4375, 0.25, 0.9375, 0.875, 1, 1, 1, 0.125, 1, 1, 1, 0.875, 1, 1, 0.25, 0.375, 0.9375, 0.25, 1, 1, 1, 0.4375, 0.5625, 0.9375, 1, 1, 0.9375, 1, 0.375, 1, 1, 1, 0.9375, 1, 1, 0.375, 1, 1, 0.6875, 1, 1, 1, 0.3125, 0.625, 1, 0.875, 1, 1, 1, 0.375, 0.875, 1, 1, 0.875, 1, 1, 0.4375, 0.8125, 1, 1, 0.8125, 1, 0.9375, 0.8125, 0.375, 1, 1, 1, 1, 0.875, 0.75, 0.6875, 1, 1, 0.75, 1, 1, 0.625, 0.75, 1, 0.875, 1, 1, 1, 0.4375, 0.5, 1, 1, 0.875, 1, 1, 0.25, 1, 1, 1, 1, 1, 1, 0.3125, 0.6875, 1, 0.9375, 0.75, 1, 0.75, 0.875, 0.8125, 1, 1, 1, 1, 0.9375, 0.625, 1, 1, 0.875, 1, 1, 1, 0.25, 0.875, 1, 0.8125, 1, 1, 1, 0.4375, 0.875, 1, 1, 0.8125, 1, 1, 0.5, 0.4375, 1, 1, 0.9375, 1, 1, 0.3125, 0.5625, 1, 0.9375, 1, 1, 1, 0.375, 0.3125, 0.5, 0.875, 1, 1, 1, 0.1875, 0.25, 1, 1, 0.8125, 0.9375, 1, 0.1875, 0.25, 0.6875, 0.75, 0.9375, 0.75, 1, 0.0625, 0.0625, 0.3125, 0.5]"

CUDA_VISIBLE_DEVICES='2' python experiments/test_after_greedy.py --model_id="meta-llama/Llama-2-7b-hf" --act_aware --ratios "[0.9375, 1, 0.8125, 0.4375, 0.375, 0.9375, 0.3125, 1, 1, 0.8125, 0.5625, 0.5, 1, 0.375, 1, 1, 1, 0.0625, 0.375, 1, 0.25, 0.8125, 1, 0.8125, 0.375, 0.5, 1, 0.5, 0.9375, 1, 0.875, 0.3125, 0.125, 0.625, 0.375, 1, 0.9375, 0.9375, 0.5, 0.375, 1, 0.375, 1, 0.875, 1, 0.5, 0.75, 0.8125, 0.6875, 0.9375, 0.9375, 0.875, 0.1875, 0.0625, 0.375, 0.4375, 1, 0.9375, 1, 0.4375, 0.5, 0.875, 0.9375, 1, 0.9375, 1, 0.4375, 0.5, 1, 0.8125, 1, 1, 0.875, 0.4375, 0.625, 0.5625, 0.75, 0.8125, 1, 1, 0.4375, 0.75, 1, 0.75, 0.9375, 1, 1, 0.375, 0.375, 0.9375, 0.9375, 1, 0.9375, 1, 0.4375, 0.5625, 1, 0.625, 0.9375, 1, 0.9375, 0.5625, 0.625, 1, 0.8125, 1, 0.875, 1, 0.375, 0.3125, 1, 0.9375, 0.75, 1, 0.9375, 0.375, 0.5625, 1, 0.9375, 0.9375, 1, 0.875, 0.375, 0.5, 1, 0.75, 1, 1, 0.75, 0.625, 0.5625, 1, 0.8125, 0.9375, 1, 0.875, 0.625, 0.5625, 1, 1, 0.5625, 0.9375, 1, 0.5625, 0.6875, 0.8125, 1, 0.875, 1, 0.875, 0.375, 0.6875, 1, 0.75, 0.8125, 1, 0.875, 0.375, 0.5625, 1, 0.8125, 1, 1, 0.875, 0.4375, 0.5, 1, 0.875, 1, 1, 0.875, 0.375, 0.4375, 1, 0.5625, 1, 1, 0.9375, 0.375, 0.375, 0.9375, 0.9375, 1, 1, 0.9375, 0.3125, 0.5625, 1, 0.625, 0.9375, 1, 1, 0.1875, 0.1875, 0.8125, 0.875, 1, 1, 0.875, 0.3125, 0.375, 1, 0.75, 0.875, 1, 1, 0.3125, 0.375, 0.875, 1, 0.6875, 0.9375, 1, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]" --split="[(1, 4), None, (4, 1), (1, 1), (1, 1), (1, 1), (1, 1), None, None, (4, 1), (1, 1), (1, 1), None, (1, 1), None, None, None, (1, 1), (1, 1), None, (1, 1), (1, 4), None, (4, 1), (1, 1), (1, 1), None, (1, 1), (1, 1), None, (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), None, (1, 1), (4, 1), (1, 1), (1, 1), None, (1, 1), None, (1, 1), None, (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), None, (1, 4), None, (1, 1), (1, 1), (1, 1), (1, 1), None, (1, 4), None, (1, 1), (1, 1), None, (1, 1), None, None, (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), None, None, (1, 1), (1, 1), None, (1, 1), (1, 4), None, None, (1, 1), (1, 1), (1, 1), (1, 1), None, (1, 4), None, (1, 1), (1, 1), None, (1, 1), (1, 4), None, (1, 1), (1, 1), (1, 1), None, (1, 1), None, (1, 1), None, (1, 1), (1, 1), None, (1, 1), (1, 1), None, (4, 1), (1, 1), (1, 1), None, (1, 1), (1, 1), None, (4, 1), (1, 1), (1, 1), None, (1, 1), None, None, (4, 1), (1, 1), (1, 1), None, (1, 1), (1, 1), None, (4, 1), (1, 1), (1, 1), None, None, (1, 1), (1, 4), None, (1, 1), (1, 1), (1, 1), None, (1, 1), None, (4, 1), (1, 1), (1, 1), None, (1, 1), (1, 1), None, (4, 1), (1, 1), (1, 1), None, (1, 1), None, None, (4, 1), (1, 1), (1, 1), None, (1, 1), None, None, (4, 1), (1, 1), (1, 1), None, (1, 1), None, None, (4, 1), (1, 1), (1, 1), (1, 1), (1, 1), None, None, (4, 1), (1, 1), (1, 1), None, (1, 1), (1, 4), None, None, (1, 1), (1, 1), (1, 1), (1, 1), None, None, (4, 1), (1, 1), (1, 1), None, (1, 1), (1, 1), None, None, (1, 1), (1, 1), (1, 1), None, (1, 1), (1, 1), None, (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]"

CUDA_VISIBLE_DEVICES='2' python experiments/test_after_greedy.py --model_id="facebook/opt-1.3b" --lora_method "UV" --act_aware --ratios "[1, 1, 0.4, 0.4, 0.8, 0.4, 0.4, 1, 1, 0.6, 1, 0.4, 0.4, 1, 0.8, 0.8, 0.8, 1, 0.6, 0.8, 0.6, 1, 0.8, 0.4, 0.8, 0.6, 0.4, 0.4, 1, 0.2, 0.4, 0.4, 0.4, 0.8, 1, 0.6, 0.6, 0.8, 1, 0.2, 0.8, 0.4, 0.6, 1, 1, 0.6, 0.6, 0.8, 1, 0.6, 1, 0.8, 0.8, 0.6, 0.8, 1, 0.8, 0.6, 1, 1, 1, 1, 0.6, 0.2, 0.8, 0.4, 1, 1, 0.6, 1, 0.6, 1, 0.6, 0.8, 1, 0.8, 0.8, 0.6, 0.4, 1, 0.8, 0.4, 0.8, 0.8, 1, 1, 0.8, 0.2, 0.8, 0.1, 1, 1, 1, 0.4, 1, 0.4, 0.4, 1, 0.8, 0.6, 0.6, 0.4, 0.8, 1, 0.8, 0.4, 0.6, 0.4, 0.6, 0.8, 0.6, 0.8, 0.4, 1, 0.1, 1, 0.8, 0.4, 0.4, 0.4, 0.6, 0.8, 0.8, 0.4, 0.2, 0.4, 0.2, 0.6, 1, 0.2, 0.1, 0.4, 0.1, 1, 0.8, 0.1, 0.2, 0.1, 0.2, 1, 1, 0.1, 0.4, 0.2, 0.4]"

CUDA_VISIBLE_DEVICES='1' python experiments/test_after_greedy.py --model_id="facebook/opt-1.3b" --lora_method "UV" --act_aware --ratios "[1, 1, 0.8, 0.4, 1, 0.4, 0.4, 1, 1, 0.8, 1, 0.8, 0.8, 1, 0.8, 1, 0.8, 1, 1, 0.8, 0.6, 1, 0.8, 0.6, 1, 0.8, 0.6, 0.4, 1, 0.6, 0.6, 0.2, 1, 1, 1, 0.4, 0.8, 1, 1, 0.4, 1, 0.2, 0.8, 1, 1, 0.6, 1, 0.8, 1, 0.6, 1, 0.6, 0.8, 0.6, 0.8, 1, 1, 0.6, 1, 0.6, 1, 1, 0.6, 0.8, 1, 0.2, 0.6, 1, 0.8, 0.6, 0.8, 0.6, 0.6, 1, 1, 0.6, 0.8, 1, 0.8, 1, 0.8, 0.8, 0.8, 0.6, 1, 1, 0.8, 0.6, 0.8, 0.6, 0.4, 1, 1, 0.4, 1, 0.4, 1, 1, 0.8, 0.6, 1, 0.6, 0.4, 0.6, 0.8, 0.4, 1, 0.4, 0.4, 0.8, 0.6, 0.4, 0.1, 0.1, 0.1, 0.6, 0.8, 0.6, 0.4, 0.8, 0.1, 0.8, 1, 0.4, 0.6, 0.4, 0.2, 0.6, 1, 0.2, 0.2, 0.4, 0.2, 0.8, 0.8, 0.1, 0.4, 0.4, 0.2, 1, 1, 0.1, 0.8, 0.4, 0.4]"

CUDA_VISIBLE_DEVICES='0' python experiments/test_after_greedy.py --model_id="facebook/opt-1.3b" --lora_method "UV"

CUDA_VISIBLE_DEVICES='2' python experiments/test_after_greedy.py --model_id="meta-llama/Llama-2-7b-hf"

# 2d sensitivity
CUDA_VISIBLE_DEVICES='0' python experiments/sensitivity.py --model_id="facebook/opt-125m" --lora_method "reconstruct" --act_aware_2d


# greedy prune

CUDA_VISIBLE_DEVICES='0' python experiments/greedy_prune.py --model_id="facebook/opt-125m" --ppl_target_st 30 --ppl_target_ed 35

CUDA_VISIBLE_DEVICES='1' python experiments/greedy_prune.py --model_id="facebook/opt-1.3b" --ppl_target_st 15.8 --ppl_target_ed 20

CUDA_VISIBLE_DEVICES='0' python experiments/greedy_prune.py --model_id="huggyllama/llama-7b" --ppl_target_st 5.8 --ppl_target_ed 8

# greedy gradient

CUDA_VISIBLE_DEVICES='0' python experiments/greedy_gradient.py --model_id="facebook/opt-125m" --ppl_target_st 33 --ppl_target_ed 40 --gradient_aware

# greedy init test

CUDA_VISIBLE_DEVICES='3' python experiments/greedy_init_test.py --model_id="meta-llama/Llama-2-7b-hf" --calib_samples 15

CUDA_VISIBLE_DEVICES='3' python experiments/greedy_init_test.py --model_id="meta-llama/Llama-2-13b-hf" --calib_samples 15


# greedy split

CUDA_VISIBLE_DEVICES='0' python experiments/greedy_split.py --model_id="facebook/opt-125m" --ppl_target_st 33 --ppl_target_ed 40 --act_aware

CUDA_VISIBLE_DEVICES='1' python experiments/greedy_split.py --model_id="facebook/opt-1.3b" --ppl_target_st 15.8 --ppl_target_ed 20 --act_aware

CUDA_VISIBLE_DEVICES='3' python experiments/greedy_split.py --model_id="huggyllama/llama-7b" --ppl_target_st 5.8 --ppl_target_ed 8 --act_aware


CUDA_VISIBLE_DEVICES='3' python experiments/greedy_split.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target_st 5.42 --ppl_target_ed 7 --act_aware

CUDA_VISIBLE_DEVICES='1' python experiments/greedy_split.py --model_id="huggyllama/llama-7b" --ppl_target_st 5.8 --ppl_target_ed 8 --act_aware --test_split 8

CUDA_VISIBLE_DEVICES='0' python experiments/greedy_split.py --model_id="huggyllama/llama-7b" --ppl_target_st 5.8 --ppl_target_ed 8 --act_aware --test_split 4


CUDA_VISIBLE_DEVICES='1' python experiments/greedy_split.py --model_id="facebook/opt-1.3b" --ppl_target_st 15.8 --ppl_target_ed 20 --act_aware

CUDA_VISIBLE_DEVICES='2' python experiments/greedy_split.py --model_id="facebook/opt-1.3b" --ppl_target_st 15.8 --ppl_target_ed 20 --act_aware

CUDA_VISIBLE_DEVICES='2' python experiments/greedy_split.py --model_id="meta-llama/Llama-2-13b-hf" --ppl_target_st 4.9 --ppl_target_ed 5.5 --act_aware

CUDA_VISIBLE_DEVICES='1' python experiments/greedy_split.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target_st 5.42 --ppl_target_ed 6.5 --act_aware

# greedy split reorder
CUDA_VISIBLE_DEVICES='0' python experiments/greedy_split.py --model_id="facebook/opt-125m" --ppl_target_st 33 --ppl_target_ed 40 --act_aware --reorder

CUDA_VISIBLE_DEVICES='0' python experiments/greedy_split.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target_st 5.42 --ppl_target_ed 7 --act_aware --reorder

CUDA_VISIBLE_DEVICES='1' python experiments/greedy_split.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target_st 5.42 --ppl_target_ed 7 --act_aware

# greedy split reorder cosearch
CUDA_VISIBLE_DEVICES='2' python experiments/greedy_split.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target_st 5.42 --ppl_target_ed 7 --act_aware --reorder --cosearch


# greedy rounded split
CUDA_VISIBLE_DEVICES='0' python experiments/greedy_iterative_split.py --model_id="facebook/opt-125m" --ppl_target_st 33 --ppl_target_ed 40 --act_aware --n_round 2

CUDA_VISIBLE_DEVICES='1' python experiments/greedy_iterative_split.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target_st 5.42 --ppl_target_ed 6.5 --act_aware --n_round 2

# greedy train scale
CUDA_VISIBLE_DEVICES='0' python experiments/greedy_train_scale.py --model_id="facebook/opt-125m" --ppl_target_st 33 --ppl_target_ed 40 --act_aware

CUDA_VISIBLE_DEVICES='1' python experiments/greedy_train_scale.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target_st 5.42 --ppl_target_ed 6.5

# greedy act full
CUDA_VISIBLE_DEVICES='0' python experiments/greedy_act_full.py --model_id="facebook/opt-125m" --ppl_target_st 33 --ppl_target_ed 40 --act_aware

CUDA_VISIBLE_DEVICES='1' python experiments/greedy_act_full.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target_st 5.42 --ppl_target_ed 6.5

# sensitivity split
CUDA_VISIBLE_DEVICES='0' python experiments/sensitivity_split.py --model_id="facebook/opt-125m" --ppl_target 40 --act_aware

CUDA_VISIBLE_DEVICES='1' python experiments/sensitivity_split.py --model_id="princeton-nlp/Sheared-LLaMA-1.3B" --ppl_target 10 --act_aware

CUDA_VISIBLE_DEVICES='2' python experiments/sensitivity_split.py --model_id="meta-llama/Llama-2-7b-hf" --ppl_target 6.5 --act_aware

CUDA_VISIBLE_DEVICES='3' python experiments/sensitivity_split.py --model_id="huggyllama/llama-7b" --ppl_target 8 --act_aware