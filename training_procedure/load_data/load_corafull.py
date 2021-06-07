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

def load_data_corafull(self):
	dataset 	= dgl.data.CoraFullDataset()
	graph 		= dataset[0]
	labels 		= [int(x) for x in graph.ndata["label"]]

	train_nodes , dev_nodes , test_nodes = self.prepare_split(labels)

	# 先去掉反向边，等会儿再加回去
	us , vs = graph.edges(order = "eid")
	m = len(us)
	graph.remove_edges( tc.LongTensor(list(range(m//2 , m))))

	labels = tc.LongTensor(labels)
	graph.ndata["feature"] = graph.ndata["feat"]
	graph = graph

	return graph , labels , train_nodes , dev_nodes , test_nodes
