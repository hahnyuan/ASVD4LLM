CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="models/gemma-2-9b" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="models/gemma-2-2b" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
