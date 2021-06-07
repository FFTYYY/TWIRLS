import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def load_sexual(self):
    V0 = pd.read_csv("./dataset/icpsr_22140/DS0001/22140-0001-Data.tsv", sep="\t", low_memory=False)
    V0 = V0[V0.STUDYNUM == 1]
    V0 = V0[(V0.SEX >= 0) & (V0.SEX <= 1)]

    E0 = pd.read_csv("./dataset/icpsr_22140/DS0002/22140-0002-Data.tsv", sep="\t", low_memory=False)
    E0 = E0[(E0.STUDYNUM == 1) & (E0.TIETYPE == 3)]

    G0 = dgl.graph(data=([], []), num_nodes=V0.shape[0])
    id2num = {id: num for num, id in enumerate(V0.RID)}

    for i, row in E0.iterrows():
        id1, id2 = row.ID1, row.ID2
        if id1 in id2num and id2 in id2num:
            G0.add_edges(id2num[id1], id2num[id2])

    G0_nx = nx.DiGraph.to_undirected(G0.to_networkx())
    lcc = list(sorted(nx.connected_components(G0_nx), key=len, reverse=True)[0])
    G = dgl.node_subgraph(G0, lcc)
    V = V0.loc[lcc]

    # print(G.number_of_nodes(), G.number_of_edges()) # FIXME: should be "1888 2096", see https://arxiv.org/pdf/2002.08274.pdf A.4

    y = 1.0 - V.SEX * 2
    y = torch.tensor(y.to_numpy(), dtype=torch.float32)

    race = F.one_hot(torch.tensor(V["RACE"].to_numpy() - 1))
    V.loc[V.BEHAV < 0, "BEHAV"] = V.BEHAV[V.BEHAV >= 0].mean()
    behav = torch.tensor(V["BEHAV"].to_numpy(), dtype=torch.float32)
    behav = (behav - behav.mean()) / behav.std()
    feature = torch.cat([race, behav.view(-1, 1)], axis=1)

    G.ndata["feature"] = feature

    perm = torch.randperm(G.number_of_nodes())
    bound_train, bound_val = int(len(perm) * 0.6), int(len(perm) * 0.8)
    train_idx, val_idx, test_idx = perm[:bound_train], perm[bound_train:bound_val], perm[bound_val:]

    return G, y, train_idx, val_idx, test_idx


if __name__ == "__main__":
    load_sexual(None)
