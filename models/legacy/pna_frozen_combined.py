from copy import deepcopy
from typing import List, Union, Callable

import dgl
import torch

from torch import nn

from models.base_layers import MLP
from models.pna import PNAGNN

class PNAFrozenCombined(nn.Module):
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
                 frozen_readout_aggregators: List[str],
                 latent3d_dim: int = 256,
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
                 **kwargs):
        super(PNAFrozenCombined, self).__init__()
        # the pretrained GNN
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
                          pretrans_layers=pretrans_layers
                          )
        self.node_gnn2D = PNAGNN(node_dim=node_dim,
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
                               pretrans_layers=pretrans_layers
                               )

        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.frozen_readout_aggregators = frozen_readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.frozen_readout_aggregators), hidden_size=latent3d_dim,
                          mid_batch_norm=False, out_dim=latent3d_dim,
                          layers=1)
        self.readout_aggregators = readout_aggregators
        self.output2D = MLP(in_dim=hidden_dim * len(self.readout_aggregators) + latent3d_dim, hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers)

    def forward(self, graph: dgl.DGLGraph):

        with torch.no_grad():
            graph3D = deepcopy(graph)
            self.node_gnn(graph3D)
            readouts_to_cat3D = [dgl.readout_nodes(graph3D, 'feat', op=aggr) for aggr in self.frozen_readout_aggregators]
            readout3D = torch.cat(readouts_to_cat3D, dim=-1)
            latent3D = self.output(readout3D).detach()

        self.node_gnn2D(graph)
        readouts_to_cat2D = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat2D + [latent3D], dim=-1)

        return self.output2D(readout)