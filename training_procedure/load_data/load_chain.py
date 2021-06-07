import torch as tc
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack , MetaApprox
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
import pdb
import dgl
import os
import pickle
import torch
import sys
import torch.optim as optim
from deeprobust.graph.utils import *
import argparse
from scipy.sparse import csr_matrix
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
import scipy
import random

def load_chain(self):

	#kth chain: i % n_chain == k
	leng = 10
	n_chain = 40
	d = 128
	label_chain = [i < n_chain // 2 for i in range(n_chain)]


	labe = [label_chain[i % n_chain] for i in range(n_chain * leng)]
	feat = tc.zeros(n_chain * leng, d)
	feat[:n_chain , 0] = tc.FloatTensor(label_chain) #给开头节点的第0维赋feature

	graph = dgl.DGLGraph()
	graph.add_nodes(n_chain * leng)

	us = list(range(n_chain , n_chain * leng)) #非头部节点
	vs = [i - n_chain for i in us] #连到上一个节点
	assert min(vs) >= 0
	graph.add_edges(us , vs)

	graph.ndata["feature"] = feat

	node_pool = set(list(range(n_chain * leng)))
	train_nodes = random.sample(node_pool , 20)

	node_pool -= set(train_nodes)
	dev_nodes = random.sample(node_pool , 100)

	node_pool -= set(dev_nodes)
	test_nodes = random.sample(node_pool , 200)

	return graph, tc.LongTensor(labe), tc.LongTensor(train_nodes), tc.LongTensor(dev_nodes), tc.LongTensor(test_nodes)


