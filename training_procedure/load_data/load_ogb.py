import pickle
from tqdm import tqdm
import torch as tc
import torch.nn as nn
import random
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from model import GNNModel
import pdb
from functools import reduce

def load_data_ogb(self):
	C = self.C
	data = DglNodePropPredDataset(name = C.data)
	graph, labels = data[0]
	splitted_idx  = data.get_idx_split()

	if C.data == "ogbn-mag": # 对于ogb-mag，只保留paper-cite部分
		graph  = dgl.edge_type_subgraph(graph , [('paper', 'cites', 'paper')])
		labels = labels["paper"]
		splitted_idx = {
			"train" : splitted_idx["train"]["paper"] ,
			"valid" : splitted_idx["valid"]["paper"] ,
			"test"  : splitted_idx["test"]["paper"] ,
		}

	train_nodes  = splitted_idx["train"]
	dev_nodes  =   splitted_idx["valid"]
	test_nodes  =  splitted_idx["test" ]

	feature = graph.ndata["feat"]
	u , v = graph.edges()
	n = graph.number_of_nodes()

	graph = dgl.DGLGraph()
	graph.add_nodes(n)
	graph.add_edges(u , v)

	labels = tc.LongTensor(labels[:,0])
	graph.ndata["feature"] = feature

	return graph , labels , train_nodes , dev_nodes , test_nodes
