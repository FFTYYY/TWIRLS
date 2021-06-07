import dgl
import numpy as np
import pandas as pd
import torch


def load_county_graph():
    # df = pd.read_csv("dataset/election/adjacency.txt", sep="\t", encoding="ISO-8859-1", header=None)
    df = pd.read_csv("./dataset/election/adjacency.txt", sep="\t", encoding="ISO-8859-1", header=None)

    hh, tt = list(df.loc[:, 1]), list(df.loc[:, 3])

    assert not np.isnan(hh[0])
    assert not np.isnan(tt).any()

    hh[0] = int(hh[0])

    for i in range(len(hh)):
        if np.isnan(hh[i]):
            hh[i] = int(hh[i - 1])
        else:
            hh[i] = int(hh[i])

    codes = list(set(hh))

    code2idx = {code: idx for idx, code in enumerate(set(tt))}
    idx2code = {idx: code for idx, code in enumerate(set(tt))}
    hh = [code2idx[h] for h in hh]
    tt = [code2idx[t] for t in tt]

    # undirected graph
    # edges = list(set(zip(hh, tt)))
    # edges_set = set(edges)

    # for u, v in edges:
    #     assert (u, v) in edges_set
    #     assert (v, u) in edges_set

    graph = dgl.graph(data=(hh, tt))

    return graph, codes


def load_features(codes, year=2012):
    assert 2011 <= year <= 2018

    df_income = pd.read_csv("./dataset/election/income.csv").set_index("FIPS")["MedianIncome" + str(year)]
    df_population = pd.read_csv("./dataset/election/population.csv").set_index("FIPS")
    df_election = pd.read_csv("./dataset/election/election.csv").set_index("fips_code")["gop_" + str(year)]
    df_education = pd.read_csv("./dataset/election/education.csv").set_index("FIPS")["BachelorRate" + str(year)]
    df_unemployment = pd.read_csv("./dataset/election/unemployment.csv").set_index("FIPS")[
        "Unemployment_rate_" + str(year)
    ]

    df_income = df_income.str.replace(",", "").astype(np.float64)
    df_migration = df_population["R_NET_MIG_" + str(year)]
    df_birth = df_population["R_birth_" + str(year)]
    df_death = df_population["R_death_" + str(year)]

    df = pd.concat([df_income, df_migration, df_birth, df_death, df_education, df_unemployment, df_election], axis=1)

    df = df.reindex(codes)

    df = df.fillna(df.mean())
    df = (df - df.mean()) / df.std()

    df = df.sort_index()

    return df


def load_county(self):
    type = self.C.data
    type_list = ["income", "migration", "birth", "death", "education", "unemployment", "election"]
    assert type in type_list

    graph, codes = load_county_graph()
    df_county = load_features(codes)
    feature = df_county.to_numpy(dtype=np.float32)

    type_idx = type_list.index(type)
    graph.ndata["feature"] = torch.tensor(np.concatenate([feature[:, :type_idx], feature[:, type_idx + 1 :]], axis=1))
    y = torch.tensor(feature[:, type_idx])

    perm = torch.randperm(graph.number_of_nodes())
    bound_train, bound_val = int(len(perm) * 0.6), int(len(perm) * 0.8)
    train_idx, val_idx, test_idx = perm[:bound_train], perm[bound_train:bound_val], perm[bound_val:]

    return graph, y, train_idx, val_idx, test_idx


if __name__ == "__main__":
    load_county("income")
