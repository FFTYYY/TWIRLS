from utils.ignn_utils import load_txt_data
import pdb
import dgl

import torch as tc

def load_amazon(self):

	self.C.multilabel = True

	portion = "0.%02d" % (self.C.train_num)
	self.logger.log("portion = %s" % portion)

	adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class = load_txt_data("amazon-all" , portion = portion)

	indices = adj.coalesce().indices()
	u,v = indices[0] , indices[1]

	graph = dgl.DGLGraph()

	graph.add_nodes(num_nodes)
	graph.add_edges(u,v)

	graph.ndata["feature"] = tc.randn(num_nodes , 16) #if no embbedding, use random feat

	return graph, labels, idx_train, idx_val, idx_test


if __name__ == "__main__":
	adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class = load_txt_data("amazon-all")

	pdb.set_trace()