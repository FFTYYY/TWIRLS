import torch as tc
from sklearn.metrics import f1_score
import torch.nn as nn
from utils.ignn_utils import Evaluation
import pdb

@tc.no_grad()
def get_eval_result(self, labels, pred_l, loss):

    if self.C.multilabel:
        micro , macro = Evaluation(pred_l , labels)
    else:
        micro = f1_score(labels.cpu(), pred_l.cpu(), average = "micro")
        macro = 0

    return {
        "micro": round(micro * 100 , 2) , # to percentage
        "macro": round(macro * 100 , 2)
    }

@tc.no_grad()
def evaluate(self, graph, all_labels, nodes_list, model, loss_func):
    
    model = model.eval()

    all_nodes = tc.cat(nodes_list)
    all_output = model(graph)[all_nodes]
    idx_from = 0
    results = []
    for nodes in nodes_list:
        idx_to = idx_from + len(nodes)
        output = all_output[idx_from:idx_to]
        idx_from = idx_to
        labels = all_labels[nodes]

        if self.C.multilabel: #multilabel
            loss = nn.BCEWithLogitsLoss()(output , labels)
            pred_l = output
        else:
            loss = loss_func(output, labels)
            pred_l = output.argmax(-1)

        results.append(get_eval_result(self, labels, pred_l, loss.item()))

    return results

