from copy import deepcopy
from typing import List, Union, Callable

import dgl
import torch

from torch import nn

from models.base_layers import MLP
from models.pna import PNAGNN

class PNAFrozen(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim,
                 target_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 readout_aggregators: List[str],
                 readout_batchnorm: bool = True,
                 readout_hidden_dim=None,
                 readout_layers: int = 2,
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 batch_norm_momentum=0.1,
                 **kwargs):
        super(PNAFrozen, self).__init__()
        self.node_gnn = PNAGNN(node_dim=node_dim,
                               edge_dim=edge_dim,
                               hidden_dim=hidden_dim,
                               aggregators=aggregators,
                               scalers=scalers,
                               residual=residual,
                               pairwise_distances=pairwise_distances,
                               activation=activation,
                               last_activation=last_activation,
                               mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm,
                               propagation_depth=propagation_depth,
                               dropout=dropout,
                               posttrans_layers=posttrans_layers,
                               pretrans_layers=pretrans_layers,
                               batch_norm_momentum=batch_norm_momentum
                               )
        self.node_gnn.requires_grad_(False)
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph: dgl.DGLGraph):
        with torch.no_grad():
            self.node_gnn(graph)
        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)
