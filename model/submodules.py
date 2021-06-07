import pdb
import random

import dgl
import torch as tc
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init
from .functions import *

class Propagate(nn.Module):
    def __init__(self):
        super().__init__()

    def prop(self, graph, Y, lam):

        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)
        Y = AX(graph, Y)
        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)

        return Y

    def forward(self, graph, Y, X, alp, lam):
        return (1 - alp) * Y + alp * lam * self.prop(graph, Y, lam) + alp * D_power_bias_X(graph, X, -1, lam, 1 - lam)


class PropagateNoPrecond(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, Y, X, alp, lam):

        return (1 - alp * lam - alp) * Y + alp * lam * normalized_AX(graph, Y) + alp * X

class Attention(nn.Module):
    def __init__(self, tau, T, p, attn_dropout = 0.0):
        super().__init__()

        self.tau = tau
        self.T = T
        self.p = p
        self.attn_dropout = attn_dropout

    def reweighting(self, graph, etas):

        w = graph.edata["w"]

        # It is not activation here but to ensure w > 0.
        # w can be < 0 here because of some precision issue in dgl, which causes NaN afterwards.
        w = F.relu(w) + 1e-7

        w = tc.pow(w, 1 - 0.5 * self.p)

        w[(w < self.tau)] = self.tau
        if self.T > 0:
            w[(w > self.T  )] = float("inf")

        w = 1 / w

        if not (w == w).all():
            raise "nan occured!"

        graph.edata["w"] = w + 1e-9 # avoid 0 degree

    def forward(self, graph, Y, etas = None):

        if etas is not None:
            Y = Y * etas.view(-1)

        # computing edge distance
        graph.srcdata["h"] = Y
        graph.srcdata["h_norm"] = (Y ** 2).sum(-1)
        graph.apply_edges(fn.u_dot_v("h", "h", "dot_"))
        graph.apply_edges(fn.u_add_v("h_norm", "h_norm", "norm_"))
        graph.edata["dot_"]  = graph.edata["dot_"].view(-1)
        graph.edata["norm_"] = graph.edata["norm_"].view(-1)
        graph.edata["w"]     = graph.edata["norm_"] - 2 * graph.edata["dot_"]
        
        # apply edge distance to get edge weight
        self.reweighting(graph, etas)

        # update node degrees
        graph.update_all(fn.copy_e("w", "m"), fn.sum("m", "deg"))
        graph.ndata["deg"] = graph.ndata["deg"].view(-1)

        # attention dropout. the implementation can ensure the degrees do not change in expectation.
        # FIXME: consider if there is a better way
        if self.attn_dropout > 0:
            graph.edata["w"] = F.dropout(graph.edata["w"], attn_dropout, training = self.training)

        return graph
