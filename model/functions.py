import pdb
import random

import dgl
import torch as tc
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init


def normalized_AX(graph, X):
    """Y = D^{-1/2}AD^{-1/2}X"""

    Y = D_power_X(graph, X, -0.5)  # Y = D^{-1/2}X
    Y = AX(graph, Y)  # Y = AD^{-1/2}X
    Y = D_power_X(graph, Y, -0.5)  # Y = D^{-1/2}AD^{-1/2}X

    return Y


def AX(graph, X):
    """Y = AX"""

    graph.srcdata["h"] = X
    graph.update_all(
        fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"),
    )
    Y = graph.dstdata["h"]

    return Y


def D_power_X(graph, X, power):
    """Y = D^{power}X"""

    degs = graph.ndata["deg"]
    norm = th.pow(degs, power)
    Y = X * norm.view(X.size(0), 1)
    return Y

def D_power_bias_X(graph, X, power, coeff, bias):
    """Y = (coeff*D + bias*I)^{power} X"""
    degs = graph.ndata["deg"]
    degs = coeff * degs + bias
    norm = th.pow(degs, power)
    Y = X * norm.view(X.size(0), 1)
    return Y

