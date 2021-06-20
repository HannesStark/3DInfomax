from typing import Union, Callable

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl.function as fn

from models.base_layers import MLP


class MPNN3D(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """
    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim,
                 target_dim,
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
        super(MPNN3D, self).__init__()
        self.node_input_net = MLP(
            in_dim=node_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            layers=1,
            mid_activation='relu',
            dropout=dropout,
            last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum

        )

        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(MPLayer(in_dim=hidden_dim,
                                           out_dim=int(hidden_dim),
                                           in_dim_edges=edge_dim,
                                           residual=residual,
                                           dropout=dropout,
                                           activation=activation,
                                           last_activation=last_activation,
                                           mid_batch_norm=mid_batch_norm,
                                           last_batch_norm=last_batch_norm,
                                           posttrans_layers=posttrans_layers,
                                           pretrans_layers=pretrans_layers,
                                           batch_norm_momentum=batch_norm_momentum
                                           ))

        self.output = MLP(in_dim=hidden_dim * 2, hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph: dgl.DGLGraph):
        graph.apply_nodes(self.input_node_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        mean_nodes = dgl.mean_nodes(graph, 'feat')
        max_nodes = dgl.max_nodes(graph, 'feat')
        mean_max = torch.cat([mean_nodes, max_nodes], dim=-1)
        return self.output(mean_max)

    def input_node_func(self, nodes):
        return {'feat': F.relu(self.node_input_net(nodes.data['feat']))}


class MPLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 in_dim_edges: int,
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 dropout: float = 0.0,
                 residual: bool = True,
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 batch_norm_momentum=0.1,
                 posttrans_layers: int = 2,
                 pretrans_layers: int = 1, ):
        super(MPLayer, self).__init__()
        self.pretrans = MLP(
            in_dim=2 * in_dim + in_dim_edges + 1,
            hidden_size=in_dim,
            out_dim=in_dim,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            layers=pretrans_layers,
            mid_activation=activation,
            dropout=dropout,
            last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum

        )
        self.posttrans = MLP(
            in_dim=2*in_dim,
            hidden_size=out_dim,
            out_dim=out_dim,
            layers=posttrans_layers,
            mid_activation=activation,
            last_activation=last_activation,
            dropout=dropout,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            batch_norm_momentum=batch_norm_momentum
        )
        self.residual = residual

    def forward(self, graph):
        h = graph.ndata['feat']
        h_in = h
        graph.update_all(message_func=self.message_function, reduce_func=fn.sum(msg='m', out='m_sum'))
        h = torch.cat([h, graph.ndata['m_sum']], dim=-1)
        # post-transformation
        h = self.posttrans(h)
        if self.residual:
            h = h + h_in
        graph.ndata['feat'] = h


    def message_function(self, edges):
        squared_distance = torch.sum((edges.src['x'] - edges.dst['x']) ** 2, dim=-1)[:, None]
        message_input = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], squared_distance], dim=-1)
        message = self.pretrans(message_input)
        return {'m': message}


