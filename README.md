
# Graph Neural Networks Inspired by Classical Iterative Algorithms

This is the code for the ICML 2021 paper ["Graph Neural Networks Inspired by Classical Iterative Algorithms"](https://arxiv.org/abs/2103.06064).

## Requirements

+ python==3.6
+ dgl>=0.6.0
+ torch_sparse

## Usage

### Datas

For amazon co-purchase data, please download the data from [IGNN's repo](https://github.com/SwiftieH/IGNN/tree/main/nodeclassification) and put them into dataset/amazon-all/. 

For Heterophily datasets, please donwload [data](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/new_data), [splits](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/splits), [structural_neighborhood](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/structural_neighborhood), and [unconnected_nodes](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/unconnected_nodes) from IGNN's repo and put them under dataset/geom_data/, dataset/splits/, dataset/structural_neighborhood/ and dataset/unconnected_nodes/ respectivly.

Other datasets would be downloaded automatically.

### Commands

Following are the commands to reproduce all the experiments in main paper.

#### Citation datasets & OGN-Arxiv

cora:
`python main.py --data=cora --mlp_bef=1 --mlp_aft=0 --dropout=0.8 --prop_step=8 --alp=1 --lam=1 --inp_dropout=0.8 --lr=0.3 --weight_decay=5e-5 --multirun=100`

citeseer:
`python main.py --mlp_bef=1 --mlp_aft=0 --prop_step=16 --lr=0.1 --num_epoch=500 --inp_dropout=0.5 --lam=1 --alp=1 --weight_decay=0.001 --multirun=100 --data=citeseer
`

pubmed:
`python main.py --mlp_bef=1 --mlp_aft=0 --prop_step=40 --lr=0.5 --num_epoch=500 --inp_dropout=0.8 --lam=1 --alp=1 --weight_decay=0.0005 --multirun=100 --data=pubmed`

ogb-arxiv:
`python main.py --data=ogbn-arxiv --lam=20 --alp=0.05 --mlp_bef=0 --mlp_aft=3 --norm=batch --hidden_size=512 --num_epoch=2000 --lr=1e-3 --prop_step=7 --dropout=0.5 --no_precond --multirun=10`

#### Dataset Under Advrsarial Attacked 
attacked cora base:
`python main.py --data=attack-struct-cora --cache_attack --mlp_bef=1 --mlp_aft=0 --inp_dropout=0.5 --prop_step=32 --num_epoch=500 --lr=0.3 --weight_decay=5e-5 --multirun=100`

attacked citeseer base:
`python main.py --data=attack-struct-citeseer --cache_attack --mlp_bef=1 --mlp_aft=0 --inp_dropout=0.5 --prop_step=64 --num_epoch=500 --lr=0.3 --weight_decay=0.001 --multirun=100`

attacked cora attention:
`python main.py --data=attack-struct-cora --cache_attack --mlp_bef=1 --mlp_aft=0 --inp_dropout=0.5 --prop_step=32 --alp=1 --lam=1 --p=0.1 --attention --tau=0.2 --num_epoch=500 --lr=0.3 --weight_decay=5e-5 --multirun=100`

attacked citeseer attention:
`python main.py --data=attack-struct-citeseer --cache_attack --mlp_bef=1 --mlp_aft=0 --inp_dropout=0.5 --prop_step=64 --alp=1 --lam=1 --p=0.1 --attention --tau=0.2 --num_epoch=500 --lr=0.3 --weight_decay=0.001 --multirun=100`

#### Heterophily Graphs

texas base:
`python main.py --data=geom-texas --multirun=10 --dropout=0 --prop_step=6 --alp=1 --lam=0.001 --lr=0.1 --weight_decay=5e-4 --hidden_size=64 --patience=200 --num_epoch=2000 --mlp_bef=2 --mlp_aft=0`

texas attention:
`python main.py --data=geom-texas --multirun=10 --dropout=0 --prop_step=6 --alp=1 --lam=0.001 --lr=0.1 --weight_decay=5e-4 --attention --attn_bef --p=0 --tau=10 --hidden_size=64 --patience=200 --num_epoch=2000 --mlp_bef=2 --mlp_aft=0`

wisconsing base:
`python main.py --data=geom-wisconsin --multirun=10 --dropout=0 --prop_step=4 --alp=1 --lam=0.001 --lr=0.5 --weight_decay=5e-4  --hidden_size=64 --patience=200 --mlp_bef=2 --mlp_aft=0`

winsconsing attention:
`python main.py --data=geom-wisconsin --multirun=10 --dropout=0 --prop_step=4 --alp=1 --lam=0.001 --lr=0.5 --weight_decay=5e-4 --attention --attn_bef --p=1 --tau=0.1 --hidden_size=64 --patience=200 --num_epoch=2000 --mlp_bef=2 --mlp_aft=0`

actor base:
`python main.py --data=geom-film --multirun=10 --dropout=0 --prop_step=4 --alp=1 --lam=0.001 --lr=0.5 --weight_decay=0.001 --hidden_size=64 --patience=200 --num_epoch=2000 --mlp_bef=2 --mlp_aft=0`

actor attention:
`python main.py --data=geom-film --multirun=10 --dropout=0 --prop_step=4 --alp=1 --lam=0.001 --lr=0.5 --weight_decay=0.001 --attention --p=1 --tau=0.01 --hidden_size=64 --patience=200 --num_epoch=2000 --mlp_bef=2 --mlp_aft=0`

cornell base:
`python main.py --data=geom-cornell --multirun=10 --dropout=0 --prop_step=4 --alp=1 --lam=0.001 --lr=0.5 --weight_decay=0.001 --hidden_size=64 --patience=200 --num_epoch=2000 --mlp_bef=2 --mlp_aft=0`

cornell attention:
`python main.py --data=geom-cornell --multirun=10 --dropout=0 --prop_step=4 --alp=1 --lam=0.001 --lr=0.5 --weight_decay=0.001 --attention --attn_bef --p=0 --tau=0.001 --hidden_size=64 --patience=200 --num_epoch=2000 --mlp_bef=2 --mlp_aft=0`

#### Long-range Denpendency

amazon co-purchase:

`python main.py --num_epoch=500 --multirun=3 --data=amazon --prop_step=32 --mlp_bef=1 --mlp_aft=0 --lam=10 --weight_decay=0 --lr=1e-2 --alp=0 --no_precond --learn_emb=128 --train_num=<label ratio> --multirun=10 --no_dev`

change \<label ratio\> to the decimal part of label ratio you want. For instance, this command use label ratio = 0.05:
`python main.py --num_epoch=500 --multirun=3 --data=amazon --prop_step=32 --mlp_bef=1 --mlp_aft=0 --lam=10 --weight_decay=0 --lr=1e-2 --alp=0 --no_precond --learn_emb=128 --train_num=5 --multirun=10 --no_dev`


