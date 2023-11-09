import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDLinear(nn.Module):
    def __init__(
        self, Us, Ss, Vs, bias=None, split="no"
    ) -> None:
        super().__init__()
        self.Us=Us
        self.Ss=Ss
        self.Vs=Vs
        if bias is not None:
            self.bias = bias
        else:
            self.bias = None
        self.split=split

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
    ):
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split==1 or oc_split==1
        if ic_split>1:
            assert linear.in_features%ic_split==0
            rank=compressed_params//(ic_split*linear.out_features+linear.in_features)
        elif oc_split>1:
            assert linear.out_features%oc_split==0
            rank=compressed_params//(oc_split*linear.in_features+linear.out_features)
        else:
            rank = compressed_params // (linear.in_features + linear.out_features)
        # print("rank", rank)
        if act_aware:
            input_abs_mean = linear.input_abs_mean
            input_abs_mean += 1e-6  # avoid zero division
            if hasattr(linear, "output_abs_mean"):
                raise NotImplementedError
                output_abs_mean = linear.output_abs_mean
                output_abs_mean += 1e-6  # avoid zero division
                input_abs_mean = input_abs_mean.sqrt()
                output_abs_mean = output_abs_mean.sqrt()
                w = (
                    linear.weight.data
                    * output_abs_mean.view(-1, 1)
                    * input_abs_mean.view(1, -1)
                )
            else:
                w = linear.weight.data * input_abs_mean.view(1, -1)
        else:
            w = linear.weight.data
        if ic_split>1:
            w=w.view(linear.out_features, ic_split, linear.in_features//ic_split)
            Us=[]
            Ss=[]
            Vs=[]
            for i in range(ic_split):
                U, S, V = torch.svd_lowrank(w[:,i,:], q=rank)
                Us.append(U)
                Ss.append(S)
                Vs.append(V)
                if act_aware:
                    V=V/input_abs_mean.view(-1,1)
            split='ic'
        elif oc_split>1:
            w=w.view(oc_split, linear.out_features//oc_split, linear.in_features)
            Us=[]
            Ss=[]
            Vs=[]
            for i in range(oc_split):
                U, S, V = torch.svd_lowrank(w[i,:,:], q=rank)
                Us.append(U)
                Ss.append(S)
                Vs.append(V)
                if act_aware:
                    V=V/input_abs_mean.view(oc_split, linear.out_features//oc_split,1)[i]
            assert act_aware==False
            split='oc'
        else:
            U, S, V = torch.svd_lowrank(w, q=rank)
            # print(S)
            Us=[U]
            Ss=[S]
            Vs=[V]
            if act_aware:
                V = V / input_abs_mean.view(-1, 1)
                if hasattr(linear, "output_abs_mean"):
                    U = U / output_abs_mean.view(-1, 1)
            split='no'

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # print shapes
        # print(f"U: {U.size()}, S: {S.size()}, V: {V.size()}, bias: {bias.size()}")
        return SVDLinear(Us, Ss, Vs, bias,split)

    def forward(self, inp):
        # compute USV^Tx + b
        if self.split in ['no','oc']:
            y=[]
            for U,S,V in zip(self.Us, self.Ss, self.Vs):
                x = F.linear(inp, V.t(), bias=None)
                x = x * S
                x = F.linear(x, U, bias=None)
                y.append(x)
            y=torch.concat(y, dim=-1)
        else:
            y=0
            if inp.dim()==2:
                inp=inp.view(inp.size(0),len(self.Us),-1)
                for i,(U,S,V) in enumerate(zip(self.Us, self.Ss, self.Vs)):
                    x = F.linear(inp[:,i,:], V.t(), bias=None)
                    x = x * S
                    x = F.linear(x, U, bias=None)
                    y+=x
            elif inp.dim()==3:
                inp=inp.view(inp.size(0),inp.size(1),len(self.Us),-1)
                for i,(U,S,V) in enumerate(zip(self.Us, self.Ss, self.Vs)):
                    x = F.linear(inp[:,:,i,:], V.t(), bias=None)
                    x = x * S
                    x = F.linear(x, U, bias=None)
                    y+=x
            else:
                raise NotImplementedError
            
        if self.bias is not None:
            y = y + self.bias
        return y

