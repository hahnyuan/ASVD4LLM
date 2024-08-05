
CUDA_VISIBLE_DEVICES='0' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.9 --use_cache --compress_kv --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='1' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.8 --use_cache --compress_kv --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.7 --use_cache --compress_kv --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='3' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.6 --use_cache --compress_kv --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='4' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.5 --use_cache --compress_kv --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='5' python asvd.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.4 --use_cache --compress_kv --eval_tasks small_longbench &



CUDA_VISIBLE_DEVICES='7' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.9 --use_cache --compress_kv --eval_tasks small_longbench
CUDA_VISIBLE_DEVICES='7' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.8 --use_cache --compress_kv --eval_tasks small_longbench 
CUDA_VISIBLE_DEVICES='7' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.7 --use_cache --compress_kv --eval_tasks small_longbench 
CUDA_VISIBLE_DEVICES='7' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.6 --use_cache --compress_kv --eval_tasks small_longbench 
CUDA_VISIBLE_DEVICES='7' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.5 --use_cache --compress_kv --eval_tasks small_longbench 
CUDA_VISIBLE_DEVICES='7' python asvd.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.4 --use_cache --compress_kv --eval_tasks small_longbench 