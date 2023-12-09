from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Llama-2-7b-hf-asvd90"
import sys

sys.path.append(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
print(model)

# model.push_to_hub(model_id)
# tokenizer.push_to_hub(model_id)
