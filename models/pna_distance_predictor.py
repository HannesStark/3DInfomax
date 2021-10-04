from typing import Dict, List, Union, Callable

import dgl
import torch
import numpy as np
from functools import partial

from torch import nn
import torch.nn.functional as F

from commons.mol_encoder import AtomEncoder, BondEncoder
from models.base_layers import MLP
from models.pna import PNAGNN


class PNADistancePredictor(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self,
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
                 projection_layers: int = 2,
                 projection_dim: int = 3,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 batch_norm_momentum=0.1,
                 **kwargs):
        super(PNADistancePredictor, self).__init__()
        self.node_gnn = PNAGNN(hidden_dim=hidden_dim, aggregators=aggregators,
                               scalers=scalers, residual=residual, pairwise_distances=pairwise_distances,
                               activation=activation, last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm, propagation_depth=propagation_depth, dropout=dropout,
                               posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                               batch_norm_momentum=batch_norm_momentum
                               )
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)
        self.distance_net = MLP(in_dim=hidden_dim * 2, hidden_size=projection_dim, mid_batch_norm=True,
                                out_dim=1, layers=projection_layers)

    def forward(self, graph: dgl.DGLGraph, pairwise_indices: torch.Tensor, mask):
        self.node_gnn(graph)
        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)

        h = graph.ndata['feat']
        src_h = torch.index_select(h, dim=0, index=pairwise_indices[0])
        dst_h = torch.index_select(h, dim=0, index=pairwise_indices[1])

        # for debugging:
        # x = graph.ndata['x']
        # src_x = torch.index_select(x, dim=0, index=pairwise_indices[0])
        # dst_x = torch.index_select(x, dim=0, index=pairwise_indices[1])

        if self.distance_net:
            src_dst_h = torch.cat([src_h, dst_h], dim=1)
            dst_src_h = torch.cat([dst_h, src_h], dim=1)
            distances = F.softplus(self.distance_net(src_dst_h) + self.distance_net(dst_src_h))
        else:
            distances = torch.norm(src_h - dst_h, dim=-1).unsqueeze(-1)

        return distances


