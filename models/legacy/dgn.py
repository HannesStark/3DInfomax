from typing import List, Union, Callable

import dgl
import torch
from goli.nn.dgl_layers.dgn_layer import DGNMessagePassingLayer
from torch import nn
import torch.nn.functional as F

from models.base_layers import MLP


class DGN(nn.Module):
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
                 **kwargs):
        super(DGN, self).__init__()
        self.node_gnn = DGNGNN(node_dim=node_dim,
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
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers)

    def forward(self, graph: dgl.DGLGraph):
        self.node_gnn(graph)
        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class DGNGNN(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 **kwargs):
        super(DGNGNN, self).__init__()
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

        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(DGNMessagePassingLayer(in_dim=hidden_dim,
                                                         out_dim=hidden_dim,
                                                         in_dim_edges=edge_dim,
                                                         aggregators=aggregators,
                                                         scalers=scalers,
                                                         dropout=dropout,
                                                         activation=activation,
                                                         last_activation=last_activation,
                                                         avg_d={"log": 1.0},
                                                         posttrans_layers=posttrans_layers,
                                                         pretrans_layers=pretrans_layers,
                                                         ),

                                  )

    def forward(self, graph: dgl.DGLGraph):
        graph.apply_nodes(self.input_node_func)
        h = graph.ndata["f"]
        ef = graph.edata["w"]
        for mp_layer in self.mp_layers:
            h_in = h
            h = mp_layer(graph, h, ef)
            h = h + h_in
        graph.ndata["f"] = h
        graph.edata["w"] = ef

    def input_node_func(self, nodes):
        return {'feat': F.relu(self.node_input_net(nodes.data['feat']))}
