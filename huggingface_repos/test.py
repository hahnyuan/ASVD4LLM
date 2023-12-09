from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM
)
import torch
import torch.nn as nn
import numpy as np

model_id="meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)

class ASVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.ALinear = nn.Linear(in_features, rank, bias=False)
        self.BLinear = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        return self.ALinear(self.BLinear(input))
    
config = model.config.to_dict()

model_dict={name:module for name,module in model.named_modules()}

config["truncation_ranks"]={}
for name,module in model.named_modules():
    if isinstance(module,nn.Linear):
        new_model=ASVDLinear(module.in_features,module.out_features,np.random.randint(100,200),bias=module.bias is not None)
        dot_pos=name.rfind(".")
        if dot_pos==-1:
            continue
        father=model_dict[name[:dot_pos]]
        setattr(father,name[dot_pos+1:],new_model)
        config["truncation_ranks"][name]=new_model.BLinear.out_features

tokenizer.save_pretrained("./llama2-7b")
model.save_pretrained("./llama2-7b")
import json

config["auto_map"]={
    "AutoConfig": "configuration_asvd_llama.ASVDLlamaConfig",
    "AutoModelForCausalLM": "modeling_asvd_llama.ASVDLlamaForCausalLM"
}
config["architectures"]=["ASVDLlamaForCausalLM"]
json.dump(config,open("./llama2-7b/config.json","w"),indent=2)

# config["vocab_size"]=100
# config["truncation_ranks"]={
#     "model.decoder.layers.0.k_proj":100,
# }
# "auto_map": {
    #   "AutoConfig": "configuration_asvd_opt.ASVDOPTConfig",
    #   "AutoModelForCausalLM": "modeling_asvd_opt.ASVDOPTForCausalLM"
    # },
# config["auto_map"]={
#     "AutoConfig": "configuration_asvd_opt.ASVDOPTConfig",
#     "AutoModelForCausalLM": "modeling_asvd_opt.ASVDOPTForCausalLM"
# }
# config["architectures"]=["ASVDOPTForCausalLM"]
# json.dump(config,open("./opt-125m-hf/config.json","w"),indent=2)

