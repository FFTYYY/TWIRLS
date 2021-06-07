from ogb.nodeproppred import DglNodePropPredDataset
import torch as tc
import torch.nn as nn
import os
import random
import dgl
from functools import reduce
from tqdm import tqdm
from model import GNNModel
from .load_data import load_data
import pdb

def prepare_train(self , model):
    '''prepare other stuffs for training'''

    C = self.C

    optimizer = tc.optim.Adam(
        params          = model.parameters() , 
        lr              = C.lr , 
        weight_decay    = C.weight_decay , 
    )
    loss_func = nn.CrossEntropyLoss(ignore_index = -100)

    return optimizer , loss_func

def prepare_model(self , label_numb , input_size , graph):
    '''create model'''

    C = self.C

    model = GNNModel(
        input_d         =  input_size , 
        output_d        =  label_numb , 
        hidden_d        =  C.hidden_size , 
        prop_step       =  C.prop_step , 
        num_mlp_before  =  C.mlp_bef , 
        num_mlp_after   =  C.mlp_aft , 
        norm            =  C.norm , 
        precond         =  not C.no_precond , 
        alp             =  C.alp , 
        lam             =  C.lam , 
        attention       =  C.attention , 
        tau             =  C.tau , 
        T               =  C.T , 
        p               =  C.p , 
        use_eta         =  C.use_eta , 
        attn_bef        =  C.attn_bef , 
        dropout         =  C.dropout , 
        attn_dropout    =  C.attn_dropout , 
        inp_dropout     =  C.inp_dropout , 
        learn_emb       =  (graph.number_of_nodes() , C.learn_emb), 
    )

    return model

def prepare_split(self , labels):
    C = self.C
    
    # whether we should change random split
    if (not self.flags.get("change split")) and (self.split_info is not None):
        return self.split_info

    n = len(labels)
    n_label = max(labels) + 1


    # generate node pool
    label_pool  = [set() for _ in range(n_label)]
    for x in set(range(n)):
        label_pool[labels[x]].add(x)
    label_pool = [s for s in label_pool if len(s) >= C.train_num + C.dev_num]

    # generate training set
    train_nodes = set()
    for label in range(len(label_pool)):
        selected = set( random.sample(label_pool[label] , C.train_num) )
        label_pool[label] -= selected
        train_nodes       |= selected

    # generate dev set
    dev_nodes = set()
    for label in range(len(label_pool)):
        selected = set( random.sample(label_pool[label] , C.dev_num) )
        label_pool[label] -= selected
        dev_nodes         |= selected

    # use remaining nodes as test tset
    test_nodes = reduce(lambda x,y : x|y , label_pool)

    self.split_info = (
        tc.LongTensor(list(train_nodes)) , 
        tc.LongTensor(list(  dev_nodes)) , 
        tc.LongTensor(list( test_nodes)) , 
    )
    return self.split_info


def init(self, idx, device = 0):
    C = self.C
    logger = self.logger

    graph , labels , train_nodes , dev_nodes , test_nodes = load_data(self, idx)
    # note if want to use rand_edge, should ensure graph is uni-directional here

    # ------------------- add random edges -------------------
    if C.rand_edge > 0:
        u , v = graph.edges()
        num_rand_edge = int(graph.number_of_edges()  * C.rand_edge)

        # decide which edges to be replaced by random edge
        rand_pos = tc.LongTensor( random.sample( range(graph.number_of_edges()) , num_rand_edge ) )

        # generate random edges
        ru = tc.randint(0 , graph.number_of_nodes()-1 , (num_rand_edge,))
        rv = tc.randint(0 , graph.number_of_nodes()-1 , (num_rand_edge,))

        # get replaced edge set
        u[rand_pos] = ru
        v[rand_pos] = rv

        graph.remove_edges(tc.arange(0,graph.number_of_edges())) # remove all edges
        graph.add_edges(u , v) # add replaced edges

        graph = graph.to_simple(copy_edata = True) # erase duplication

    # ------------------- turn to bidirectional -------------------
    u , v = graph.edges(order = "eid")
    graph.add_edges(v , u) # bidirect
    graph = graph.to_simple(copy_edata = True) # erase duplication

    # ------------------- add self-loops -------------------
    graph = graph.remove_self_loop()
    graph = graph.   add_self_loop()
    logger.log("train / dev / test = %d / %d / %d" % (len(train_nodes) , len(dev_nodes) , len(test_nodes)))

    # ------------------- prepare model -------------------
    label_numb = int(labels.size(-1)) if C.multilabel else int(max(labels)) + 1
    input_size = graph.ndata["feature"].size(-1)
    model = self.prepare_model(label_numb , input_size , graph)

    logger.log("number of params: %d" % sum( [int(x.view(-1).size(0)) for x in model.parameters()] ))

    optimizer , loss_func = self.prepare_train(model)

    # ------------------- move evrything to gpu -------------------
    graph       = graph      .to(device)
    labels      = labels     .to(device)
    train_nodes = train_nodes.to(device)
    dev_nodes   = dev_nodes  .to(device)
    test_nodes  = test_nodes .to(device)
    model       = model      .to(device)
    return (graph , labels) , (train_nodes , dev_nodes , test_nodes) , model , (optimizer , loss_func)