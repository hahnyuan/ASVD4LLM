CUDA_VISIBLE_DEVICES='0' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.9 --use_cache --seed 42 --eval_tasks mmlu
CUDA_VISIBLE_DEVICES='1' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.8 --use_cache --seed 42 --eval_tasks mmlu
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.7 --use_cache --seed 42 --eval_tasks mmlu

CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.9 --use_cache --seed 42 --eval_tasks mmlu
CUDA_VISIBLE_DEVICES='4' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.8 --use_cache --seed 42 --eval_tasks mmlu
CUDA_VISIBLE_DEVICES='5' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.7 --use_cache --seed 42 --eval_tasks mmlu

CUDA_VISIBLE_DEVICES='6' python asvd_cholesky.py --model_id="models/Llama-3.1-8b" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.9 --use_cache --seed 42 --eval_tasks mmlu



CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.9 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.85 --use_cache --calib_dataset selfgen --seed 42

CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.9 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.85 --use_cache --calib_dataset selfgen --seed 42

CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.9 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.85 --use_cache --calib_dataset c4 --seed 42

CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.9 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.85 --use_cache --calib_dataset c4 --seed 42

# DEBUG
CUDA_VISIBLE_DEVICES='0' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='1' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='1' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='2' python -m pdb asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='3' python -m pdb asvd_cholesky.py --model_id="models/Llama-3.1-8b" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='0' python -m pdb asvd_cholesky.py --model_id="facebook/opt-125m" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --eval_tasks mmlu
CUDA_VISIBLE_DEVICES='0' python -m pdb asvd_cholesky.py --model_id="facebook/opt-125m" --n_calib_samples 32 --transform_mat_method magnitude_alpha0.5 --param_ratio_target 0.95 --use_cache --eval_tasks mmlu

lm_eval --model hf --model_args pretrained=models/Llama-2-7b-hf --tasks mmlu --device cuda:0 --batch_size auto

CUDA_VISIBLE_DEVICES='0' python asvd_cholesky.py --model_id="facebook/opt-125m" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --eval_tasks mmlu --no_svd

CUDA_VISIBLE_DEVICES='1' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.9
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="facebook/opt-125m" --n_calib_samples 32 --transform_mat_method magnitude_alpha0.5 --param_ratio_target 0.9
