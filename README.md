# Run
CUDA_VISIBLE_DEVICES='0,1' python svd_lora_train.py --model_id="huggyllama/llama-7b" --saved_dir="./output/llama-7b-svdlora" --using_svd --rank_compress_ratio=0.2
CUDA_VISIBLE_DEVICES='1' python svd_lora_train.py --model_id="facebook/opt-125m" --saved_dir="./output/opt-125m-svdlora" --using_svd --rank_compress_ratio=0.2