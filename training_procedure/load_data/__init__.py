import pdb
import pickle
import random
from functools import reduce

import dgl
import torch as tc
import torch.nn as nn
from model import GNNModel
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm

from .load_attacked_citation import load_attacked_citation
from .load_citation import load_data_citation
from .load_corafull import load_data_corafull
from .load_county import load_county
from .load_geom import load_data_geom
from .load_ogb import load_data_ogb
from .load_sexual import load_sexual
from .load_chain import load_chain
from .load_amazon import load_amazon

"""
load_datas_xxx return:
    graph , labels , train_nodes , dev_nodes , test_nodes


graph: dgl.graph
graph.ndata["feature"]: torch.FloatTensor(n , inp_d)

labels: torch.LongTensor(n)
train_nodes: torch.LongTensor(n)
dev_nodes: torch.LongTensor(n)
test_nodes: torch.LongTensor(n)

"""


def load_data(self, idx):
    C = self.C
    C.multilabel = False
    
    if C.data == "cora-full":
        graph, labels, train_nodes, dev_nodes, test_nodes = load_data_corafull(self)
    elif C.data.startswith("ogbn"):
        graph, labels, train_nodes, dev_nodes, test_nodes = load_data_ogb(self)
    elif C.data in ["cora", "citeseer", "pubmed"]:
        graph, labels, train_nodes, dev_nodes, test_nodes = load_data_citation(self)
    elif C.data.startswith("geom"):
        graph, labels, train_nodes, dev_nodes, test_nodes = load_data_geom(self, idx)
    elif C.data == "chain":
        graph, labels, train_nodes, dev_nodes, test_nodes = load_chain(self)
    elif C.data.startswith("attack-struct-"): #attack-struct-cora , attack-struct-citeceer
        graph, labels, train_nodes, dev_nodes, test_nodes = load_attacked_citation(self)
    elif C.data in ["income", "education", "unemployment", "election"]:
        graph, labels, train_nodes, dev_nodes, test_nodes = load_county(self)
    elif C.data == "ising":
        ...
    elif C.data == "anaheim":
        ...
    elif C.data == "chacago":
        ...
    elif C.data == "sexual":
        graph, labels, train_nodes, dev_nodes, test_nodes = load_sexual(self)
    elif C.data == "twitch-pt":
        ...
    elif C.data == "amazon":
        graph, labels, train_nodes, dev_nodes, test_nodes = load_amazon(self)
    else:
        assert False

    return graph, labels, train_nodes, dev_nodes, test_nodes

