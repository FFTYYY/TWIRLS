import pdb
import random
from functools import partial
import dgl
import torch as tc
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init
from .submodules import Propagate, PropagateNoPrecond , Attention

class UnfoldindAndAttention(nn.Module):
    def __init__(self, d, alp, lam, prop_step, attn_aft, tau, T, p, use_eta, init_att , attn_dropout, precond):

        super().__init__()

        self.d      = d
        self.alp    = alp if alp > 0 else 1 / (lam + 1) # automatic set alpha
        self.lam    = lam
        self.tau    = tau
        self.p      = p
        self.prop_step = prop_step
        self.attn_aft  = attn_aft
        self.use_eta   = use_eta
        self.init_att  = init_att

        prop_method      = Propagate if precond else PropagateNoPrecond
        self.prop_layers = nn.ModuleList([prop_method() for _ in range(prop_step)])

        self.init_attn   = Attention(tau, T, p, attn_dropout) if self.init_att      else None
        self.attn_layer  = Attention(tau, T, p, attn_dropout) if self.attn_aft >= 0 else None
        self.etas        = nn.Parameter(tc.ones(d)) if self.use_eta else None

    def forward(self , g , X):
        
        Y = X


        g.edata["w"]    = tc.ones(g.number_of_edges(), 1, device = g.device)
        g.ndata["deg"]  = g.in_degrees().float()

        if self.init_att:
            g = self.init_attn(g, Y, self.etas)

        for k, layer in enumerate(self.prop_layers):

            # do unfolding
            Y = layer(g, Y, X, self.alp, self.lam)

            # do attention at certain layer
            if k == self.attn_aft - 1:
                g = self.attn_layer(g, Y, self.etas)

        return Y

class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, num_layers, dropout, norm, init_activate) :
        super().__init__()

        self.init_activate  = init_activate
        self.norm           = norm
        self.dropout        = dropout

        self.layers = nn.ModuleList([])

        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for k in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))

        self.norm_cnt = num_layers-1+int(init_activate) # how many norm layers we have
        if norm == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_d) for _ in range(self.norm_cnt)])
        elif norm == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm  (hidden_d) for _ in range(self.norm_cnt)])


        self.reset_params()

    def reset_params(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.constant_     (layer.bias.data, 0)

    def activate(self, x):
        if self.norm != "none":
            x = self.norms[self.cur_norm_idx](x) # use the last norm layer
            self.cur_norm_idx += 1
        x = F.relu(x)
        x = F.dropout(x , self.dropout , training = self.training)
        return x 


    def forward(self, x):
        self.cur_norm_idx = 0

        if self.init_activate:
            x = self.activate(x)

        for i , layer in enumerate( self.layers ):
            x = layer(x)
            if i != len(self.layers) - 1: # do not activate in the last layer
                x = self.activate(x)

        return x
