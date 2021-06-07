import pdb

import torch as tc
import torch.nn as nn


def train(self, graph, labels, nodes, model, loss_func, optimizer):

    model = model.train()

    output = model(graph)[nodes]
    labels = labels[nodes]

    if self.C.multilabel: # use BCEWithLogitsLoss in multilabel setting
        loss = nn.BCEWithLogitsLoss()(output , labels)
    else:
        loss = loss_func(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model, float(loss)
