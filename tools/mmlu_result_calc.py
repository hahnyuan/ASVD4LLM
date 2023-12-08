import os

file="output/result.txt"

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
            print(rst["ptb"],"ptb")
            print(rst["wikitext2"],"wikitext2")
        else:
            print(l)