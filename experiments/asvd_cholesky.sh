CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.9 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.85 --use_cache --calib_dataset selfgen --seed 42

CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.95 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.9 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.85 --use_cache --calib_dataset selfgen --seed 42

CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.95 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.9 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.85 --use_cache --calib_dataset c4 --seed 42

CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.95 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.9 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.85 --use_cache --calib_dataset c4 --seed 42

# DEBUG
CUDA_VISIBLE_DEVICES='2' python -m pdb asvd_cholesky.py --model_id="models/Llama-2-7b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.95 --use_cache --seed 42 --cholesky=cache/cholesky_models_Llama_2_7b_hf_c4_32_42.pt
CUDA_VISIBLE_DEVICES='1' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='1' python asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='2' python -m pdb asvd_cholesky.py --model_id="models/Llama-2-13b-hf" --act_aware --n_calib_samples 32 --scaling_method whitening --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='3' python -m pdb asvd_cholesky.py --model_id="models/Llama-3.1-8b" --act_aware --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --seed 42
CUDA_VISIBLE_DEVICES='4' python -m pdb asvd_cholesky.py --model_id="facebook/opt-125m" --act_aware --n_calib_samples 32 --transform_mat_method cholesky --param_ratio_target 0.95 --use_cache --seed 42
