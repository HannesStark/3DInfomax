from typing import Dict, List, Union, Callable

import dgl
import torch
import numpy as np
from functools import partial

from torch import nn
import torch.nn.functional as F

from models.base_layers import MLP
from models.pna import PNA_AGGREGATORS, PNA_SCALERS


class PNAEGNN(nn.Module):
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
                 pretrans_complete_layers=2,
                 batch_norm_momentum=0.1,
                 **kwargs):
        super(PNAEGNN, self).__init__()
        self.node_gnn = PNAEGNNNet(node_dim=node_dim,
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
                                   pretrans_complete_layers=pretrans_complete_layers,
                                   batch_norm_momentum=batch_norm_momentum
                                   )
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph: dgl.DGLHeteroGraph):
        self.node_gnn(graph)
        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class PNAEGNNNet(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 batch_norm_momentum=0.1,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 pretrans_complete_layers=2,
                 **kwargs):
        super(PNAEGNNNet, self).__init__()
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
                batch_norm_momentum=batch_norm_momentum
            )
        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(PNAEGNNLayer(in_dim=hidden_dim,
                                               out_dim=int(hidden_dim),
                                               in_dim_edges=hidden_dim,
                                               aggregators=aggregators,
                                               scalers=scalers,
                                               pairwise_distances=pairwise_distances,
                                               residual=residual,
                                               dropout=dropout,
                                               activation=activation,
                                               last_activation=last_activation,
                                               mid_batch_norm=mid_batch_norm,
                                               last_batch_norm=last_batch_norm,
                                               avg_d={"log": 1.0},
                                               posttrans_layers=posttrans_layers,
                                               pretrans_layers=pretrans_layers,
                                               pretrans_complete_layers=pretrans_complete_layers,
                                               batch_norm_momentum=batch_norm_momentum
                                               ),

                                  )
        self.node_wise_output_network = MLP(
            in_dim=hidden_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=2,
            mid_activation=activation,
            dropout=dropout,
            last_activation='None',
        )

    def forward(self, graph: dgl.DGLGraph):
        graph.apply_nodes(self.input_node_func)
        graph.apply_edges(self.input_edge_func, etype='bond')

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        graph.apply_nodes(self.output_node_func)

    def input_node_func(self, nodes):
        return {'feat': F.relu(self.node_input_net(nodes.data['feat']))}

    def input_edge_func(self, edges):
        return {'feat': F.relu(self.edge_input(edges.data['feat']))}

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}


class PNAEGNNLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 in_dim_edges: int,
                 aggregators: List[str],
                 scalers: List[str],
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 dropout: float = 0.0,
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 batch_norm_momentum=0.1,
                 avg_d: Dict[str, float] = {"log": 1.0},
                 posttrans_layers: int = 2,
                 pretrans_layers: int = 1,
                 pretrans_complete_layers: int = 2):
        super(PNAEGNNLayer, self).__init__()
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MLP(
            in_dim=2 * in_dim + in_dim_edges,
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
            in_dim=(len(self.aggregators) * len(self.scalers) * 2 + 1) * in_dim,
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
        self.soft_edge_network = nn.Linear(out_dim, 1)

        self.pretrans_complete = MLP(
            in_dim=in_dim * 2,
            hidden_size=in_dim,
            out_dim=in_dim,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            layers=pretrans_complete_layers,
            mid_activation=activation,
            dropout=dropout,
            last_activation='None',
            batch_norm_momentum=batch_norm_momentum
        )

    def forward(self, g):
        h = g.ndata['feat']
        h_in = h
        # pretransformation
        g.apply_edges(self.pretrans_edges, etype='bond')
        g.apply_edges(self.pretrans_complete_edges, etype='complete')

        # aggregation
        g.update_all(self.message_func, self.reduce_func, etype='bond')
        g.update_all(self.message_func_complete, self.reduce_func_complete, etype='complete')

        h = torch.cat([h, g.ndata['f_bond'], g.ndata['f_complete']], dim=-1)
        # post-transformation
        h = self.posttrans(h)
        if self.residual:
            h = h + h_in

        g.ndata['feat'] = h

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {"e": edges.data["e"]}

    def message_func_complete(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {"e_complete": edges.data["e_complete"]}

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data['feat']
        h = nodes.mailbox["e"]
        D = h.shape[-2]
        h_to_cat = [aggr(h=h, h_in=h_in) for aggr in self.aggregators]
        h = torch.cat(h_to_cat, dim=-1)

        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=-1)

        return {'f_bond': h}

    def reduce_func_complete(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data['feat']
        h = nodes.mailbox["e_complete"]
        D = h.shape[-2]
        h_to_cat = [aggr(h=h, h_in=h_in) for aggr in self.aggregators]
        h = torch.cat(h_to_cat, dim=-1)

        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=-1)

        return {'f_complete': h}

    def pretrans_edges(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).
        """
        z2 = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=-1)
        return {"e": self.pretrans(z2)}

    def pretrans_complete_edges(self, edges):
        message_input = torch.cat([edges.src['feat'], edges.dst['feat']], dim=-1)
        message = self.pretrans_complete(message_input)
        edge_weight = torch.sigmoid(self.soft_edge_network(message))
        return {'e_complete': message * edge_weight}
