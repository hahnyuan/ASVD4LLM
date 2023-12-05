import os

file="output/final/meta-llama_Llama-2-7b-hf.json"
# file="output/final/huggyllama_llama-7b.json"

with open(file) as f:
    lines=f.readlines()
    for l in lines:
        if l[0]=='{' and '}' in l[-2:]:
            # print(l)
            rst=eval(l)
            mean_acc=[]
            for k,v in rst.items():
                if isinstance(v,dict):
                    acc=v['acc']
                    mean_acc.append(acc)
            if len(mean_acc)!=0:
                print(f"{sum(mean_acc)/len(mean_acc):.4f}")
        else:
            print(l)