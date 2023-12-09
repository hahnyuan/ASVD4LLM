from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
sys.path.append('.')

model_id = "hahnyuan/opt-125m-asvd90"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, 
)
print(model)

from evaluate import evaluate_model
evaluate_model(model, tokenizer, model_id, "", eval_ppl="wikitext2,ptb", limit=-1)