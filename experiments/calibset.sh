CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --calib_dataset selfgen --seed 42

CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --calib_dataset selfgen --seed 42

CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --calib_dataset c4 --seed 42

CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --calib_dataset c4 --seed 42
