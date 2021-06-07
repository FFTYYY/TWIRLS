import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import dgl
from model.modules import UnfoldindAndAttention , MLP
import pdb

class GNNModel(nn.Module):
    def __init__( self , 
        input_d     , 
        output_d    , 
        hidden_d    , 
        prop_step   , 
        num_mlp_before = 1,
        num_mlp_after  = 1, 
        norm        = 'none' , 
        precond     = True  ,
        alp         = 0     , 
        lam         = 1     , 
        attention   = False , 
        tau         = 0.2   , 
        T           = -1    , 
        p           = 1     , 
        use_eta     = False ,
        attn_bef    = False , 
        dropout     = 0.0   ,
        attn_dropout= 0.0   , 
        inp_dropout = 0.0   ,
        learn_emb   = (0,0) , 
    ):
        super().__init__()
        self.input_d        = input_d
        self.output_d       = output_d
        self.hidden_d       = hidden_d
        self.prop_step      = prop_step
        self.num_mlp_before = num_mlp_before
        self.num_mlp_after  = num_mlp_after
        self.norm           = norm
        self.precond        = precond
        self.attention      = attention
        self.alp            = alp
        self.lam            = lam
        self.tau            = tau
        self.T              = T
        self.p              = p
        self.use_eta        = use_eta
        self.init_att       = attn_bef
        self.dropout        = dropout
        self.attn_dropout   = attn_dropout
        self.inp_dropout    = inp_dropout
        self.learn_emb      = learn_emb

        # ----- initialization of some variables -----
        # where to put attention
        self.attn_aft = prop_step // 2 if attention else -1 

        # whether to learn a embedding for each node. used in amazon co-purchase dataset
        if self.learn_emb[1] > 0:
            self.node_emb = nn.Parameter(tc.randn(learn_emb[0] , learn_emb[1]))
            self.input_d  = learn_emb[1]
            nn.init.normal_(self.node_emb , 0 , 1e-3)
            
        # whether we can cache unfolding result
        self.cacheable = (not self.attention) and self.num_mlp_before == 0 and self.inp_dropout <= 0 and self.learn_emb[1] <= 0
        if self.cacheable:
            self.cached_unfolding = None


        # if only one layer, then no hidden size
        self.size_bef_unf = self.hidden_d
        self.size_aft_unf = self.hidden_d
        if self.num_mlp_before == 0:
            self.size_aft_unf = self.input_d  # as the input  of mlp_aft
        if self.num_mlp_after == 0:
            self.size_bef_unf = self.output_d # as the output of mlp_bef


        # ----- computational modules -----
        self.mlp_bef = MLP(self.input_d , self.hidden_d , self.size_bef_unf , self.num_mlp_before , 
                self.dropout , self.norm , init_activate = False)

        self.unfolding = UnfoldindAndAttention(self.hidden_d, self.alp, self.lam, self.prop_step, self.attn_aft, 
                self.tau, self.T, self.p, self.use_eta, self.init_att, self.attn_dropout, self.precond)

        # if there are really transformations before unfolding, then do init_activate in mlp_aft
        self.mlp_aft = MLP(self.size_aft_unf , self.hidden_d , self.output_d , self.num_mlp_after  , 
            self.dropout , self.norm , 
            init_activate = (self.num_mlp_before > 0) and (self.num_mlp_after > 0) 
        )

    def forward(self , g):

         # use trained node embedding
        if self.learn_emb[1] > 0:
            x = self.node_emb
        else:
            x = g.ndata["feature"]

        if self.cacheable: 
            # to cache unfolding result becase there is no paramaters before it
            if self.cached_unfolding is None:
                self.cached_unfolding = self.unfolding(g , x)

            x = self.cached_unfolding
        else:
            if self.inp_dropout > 0:
                x = F.dropout(x, self.inp_dropout, training = self.training)
            x = self.mlp_bef(x)
            x = self.unfolding(g , x)

        x = self.mlp_aft(x)

        return x




