from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import torch as th


def load_data_citation(self):
	if self.C.data == 'cora':
		dataset = CoraGraphDataset()
	if self.C.data == 'citeseer':
		dataset = CiteseerGraphDataset()
	if self.C.data == 'pubmed':
		dataset = PubmedGraphDataset()
	graph = dataset[0]
	# 删除自环
	graph = graph.remove_self_loop()
	graph.ndata['feature'] = graph.ndata['feat']
	labels = graph.ndata['label'].view(-1)
	train_nodes = th.where(graph.ndata['train_mask'] == True)[0]
	dev_nodes = th.where(graph.ndata['val_mask'] == True)[0]
	test_nodes = th.where(graph.ndata['test_mask'] == True)[0]
	# 变为单向图
	remove_reverse_edges(graph)
	return graph, labels, train_nodes, dev_nodes, test_nodes


def remove_reverse_edges(graph):
	# 将双向边变为单向边
	edges = graph.edges()
	m = graph.num_edges()
	edges_set = set()
	remove_eids = []
	for i in range(m):
		u, v = edges[0][i].item(), edges[1][i].item()
		if u > v:
			e = (v, u)
		else:
			e = (u, v)
		if e in edges_set:
			remove_eids.append(i)
		else:
			edges_set.add(e)
	graph.remove_edges(th.LongTensor(remove_eids))
