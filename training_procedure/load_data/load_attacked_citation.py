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

mem_datas = None # even don't cache on disk
tmp_save_file = "dataset/attacked_cache/att_mettack_struct_{name}.pkl"
ptb_rate = 0.2

def to_cpu(mem_datas):
	graph, labels, train_nodes, dev_nodes, test_nodes = mem_datas
	graph 		= graph.to("cpu")
	labels 		= labels.to("cpu")
	train_nodes = train_nodes.to("cpu")
	dev_nodes 	= dev_nodes.to("cpu")
	test_nodes 	= test_nodes.to("cpu")
	return graph, labels, train_nodes, dev_nodes, test_nodes


def load_attacked_citation(self):
	'''note, this function output symmetric adj, so never use --rand_edge again'''

	global mem_datas
	if mem_datas is not None:

		return to_cpu(mem_datas)

	dataset_name = self.C.data.split("-")[-1]
	tmp_save_file_name = tmp_save_file.format(name = dataset_name)

	if dataset_name == "pubmed":
		device = "cpu"
	else:
		device = 0

	os.makedirs(os.path.dirname(tmp_save_file_name) , exist_ok = True)
	if os.path.exists(tmp_save_file_name) and self.C.cache_attack: #使用保存的图
		with open(tmp_save_file_name , "rb") as fil:
			mem_datas = pickle.load(fil)
		self.logger.log("use cached attack data : %s" % tmp_save_file_name)
		return to_cpu(mem_datas)

	# --- exactly the same setting with GNNGuard ---
	data = Dataset(root='/tmp/', name = dataset_name)

	# read adj & features
	adj, features, labels = data.adj, data.features, data.labels
	idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
	idx_unlabeled = np.union1d(idx_val, idx_test)
	if scipy.sparse.issparse(features) == False:
		features = scipy.sparse.csr_matrix(features)
	perturbations = int(ptb_rate * (adj.sum()//2))
	adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
	adj, features = csr_matrix(adj), csr_matrix(features)

	# to bidirected
	adj = adj + adj.T
	adj[adj>1] = 1

	# Setup GCN as the Surrogate Model
	surrogate = GCN(nfeat = features.shape[1], nclass = labels.max().item()+1, nhid = 16,
			dropout = 0.5, with_relu = False, with_bias = True, weight_decay = 5e-4, device = device).to(device)
	surrogate.fit(features, adj, labels, idx_train, train_iters=201)

	# Mettack - Self
	model = Metattack(model = surrogate, nnodes = adj.shape[0], feature_shape = features.shape,  
		attack_structure = True, attack_features = False, device = device, lambda_ = 0.)
	model = model.to(device)

	# run attack
	model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)

	# convert to pytorch dense
	modified_adj = model.modified_adj
	features = sparse_mx_to_torch_sparse_tensor(features).to_dense()

	# --- ---

	fea = features.cpu()
	adj = modified_adj.cpu()

	n_nodes = fea.shape[0]

	graph = dgl.DGLGraph()
	graph.add_nodes(n_nodes)

	us , vs = tc.where(adj > 0)
	graph.add_edges(us , vs)

	graph.ndata["feature"] = tc.FloatTensor(fea)
	labels = tc.LongTensor(labels).view(-1).cuda()
	train_nodes = tc.LongTensor(idx_train).view(-1).cuda()
	dev_nodes   = tc.LongTensor(idx_val).view(-1).cuda()
	test_nodes  = tc.LongTensor(idx_test).view(-1).cuda()

	graph = graph.to(0)


	mem_datas = (graph, labels, train_nodes, dev_nodes, test_nodes)

	# cache data
	with open(tmp_save_file_name , "wb") as fil:
		pickle.dump(mem_datas , fil)

	return to_cpu(mem_datas)


if __name__ == "__main__":
	load_attacked_cora(None)


