from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from models.base_layers import MLP


class EGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, target_dim, readout_aggregators: List[str], batch_norm=False,
                 readout_batchnorm=True,
                 dropout=0.0, propagation_depth: int = 4, readout_layers: int = 2, readout_hidden_dim=None,
                 mid_activation: str = 'SiLU', **kwargs):
        super(EGNN, self).__init__()
        self.input = MLP(
            in_dim=node_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=1,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )
        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(EGCLayer(node_dim, hidden_dim=hidden_dim, batch_norm=batch_norm, dropout=dropout,
                                           mid_activation=mid_activation))

        self.node_wise_output_network = MLP(
            in_dim=hidden_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers)

    def forward(self, graph: dgl.DGLGraph):
        graph.apply_nodes(self.input_node_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [dgl.readout_nodes(graph, 'f', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)

    def output_node_func(self, nodes):
        return {'f': self.node_wise_output_network(nodes.data['f'])}

    def input_node_func(self, nodes):
        return {'f': F.silu(self.input(nodes.data['f']))}

    def input_edge_func(self, edges):
        return {'w': F.silu(self.edge_input(edges.data['w']))}


class EGCLayer(nn.Module):
    def __init__(self, node_dim, hidden_dim, batch_norm, dropout, mid_activation):
        super(EGCLayer, self).__init__()
        self.message_network = MLP(
            in_dim=hidden_dim * 2 + 1,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation=mid_activation,
        )

        self.update_network = MLP(
            in_dim=hidden_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )

        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=fn.sum(msg='m', out='m_sum'),
                         apply_node_func=self.update_function)

    def message_function(self, edges):
        squared_distance = edges.data['w'] ** 2
        message_input = torch.cat(
            [edges.src['f'], edges.dst['f'], squared_distance], dim=-1)
        message = self.message_network(message_input)
        edge_weight = torch.sigmoid(message)
        return {'m': message * edge_weight}

    def update_function(self, nodes):
        h = nodes.data['f']
        input = torch.cat([nodes.data['m_sum'] + nodes.data['f']], dim=-1)
        h_new = self.update_network(input)
        output = h_new + h
        return {'f': output}
