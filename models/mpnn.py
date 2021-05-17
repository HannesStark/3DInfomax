from typing import Callable, List, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from models.base_layers import MLP


class MPNN(nn.Module):
    """
    Message Passing Neural Network
    """

    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim,
                 target_dim,
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
                 **kwargs):
        super(MPNN, self).__init__()
        self.node_gnn = MPNNGNN(node_dim=node_dim,
                          edge_dim=edge_dim,
                          hidden_dim=hidden_dim,
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
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers)

    def forward(self, graph: dgl.DGLGraph):
        self.node_gnn(graph)
        readouts_to_cat = [dgl.readout_nodes(graph, 'f', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)

class MPNNGNN(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim,
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
        super(MPNNGNN, self).__init__()
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
        )
        if edge_dim > 0:
            self.edge_input = MLP(
                in_dim=edge_dim,
                hidden_size=hidden_dim,
                out_dim=hidden_dim,
                mid_batch_norm=mid_batch_norm,
                last_batch_norm=last_batch_norm,
                layers=1,
                mid_activation='relu',
                dropout=dropout,
                last_activation=last_activation,
            )
        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(MPNNLayer(in_dim=hidden_dim,
                                           out_dim=int(hidden_dim),
                                           in_dim_edges=edge_dim,
                                           pairwise_distances=pairwise_distances,
                                           residual=residual,
                                           dropout=dropout,
                                           activation=activation,
                                           last_activation=last_activation,
                                           mid_batch_norm=mid_batch_norm,
                                           last_batch_norm=last_batch_norm,
                                           posttrans_layers=posttrans_layers,
                                           pretrans_layers=pretrans_layers,
                                           ),

                                  )

    def forward(self, graph: dgl.DGLGraph):
        graph.apply_nodes(self.input_node_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

    def input_node_func(self, nodes):
        return {'f': F.relu(self.node_input_net(nodes.data['f']))}

    def input_edge_func(self, edges):
        return {'w': F.relu(self.edge_input(edges.data['w']))}


class MPNNLayer(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 in_dim_edges: int,
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 dropout: float = 0.0,
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 posttrans_layers: int = 2,
                 pretrans_layers: int = 1, ):
        super(MPNNLayer, self).__init__()

        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MLP(
            in_dim=(2 * in_dim + in_dim_edges + 1) if self.pairwise_distances else (2 * in_dim + in_dim_edges),
            hidden_size=in_dim,
            out_dim=in_dim,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            layers=pretrans_layers,
            mid_activation='relu',
            dropout=dropout,
            last_activation=last_activation,
        )
        self.posttrans = MLP(
            in_dim=in_dim,
            hidden_size=out_dim,
            out_dim=out_dim,
            layers=posttrans_layers,
            mid_activation=self.activation,
            last_activation=last_activation,
            dropout=dropout,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
        )


    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=fn.sum(msg='m', out='m_sum'))
        graph.apply_nodes(self.update_function)

    def message_function(self, edges):
        if self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x']) ** 2, dim=-1)[:, None]
            message_input = torch.cat([edges.src['f'], edges.dst['f'], edges.data['w'], squared_distance], dim=-1)
        else:
            message_input = torch.cat([edges.src['f'], edges.dst['f'], edges.data['w']], dim=-1)
        message = self.pretrans(message_input)
        return {'m': message}

    def update_function(self, nodes):
        return {'f': self.posttrans(nodes.data['m_sum'] + nodes.data['f'])}




