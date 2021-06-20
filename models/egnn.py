from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from commons.utils import fourier_encode_dist
from models.base_layers import MLP


class EGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, target_dim, readout_aggregators: List[str], batch_norm=False,
                 readout_batchnorm=True
                 , batch_norm_momentum=0.1, reduce_func='sum',
                 dropout=0.0, propagation_depth: int = 4, readout_layers: int = 2, readout_hidden_dim=None,
                 fourier_encodings=0,
                 mid_activation: str = 'SiLU', **kwargs):
        super(EGNN, self).__init__()
        self.fourier_encodings = fourier_encodings
        self.input = MLP(
            in_dim=node_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum = batch_norm_momentum,
            layers=1,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )

        self.mp_layers = nn.ModuleList()
        edge_in_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1
        for _ in range(propagation_depth):
            self.mp_layers.append(
                EGCLayer(node_dim, edge_dim=edge_in_dim, hidden_dim=hidden_dim, batch_norm=batch_norm, batch_norm_momentum = batch_norm_momentum, dropout=dropout,
                         mid_activation=mid_activation, reduce_func=reduce_func))

        self.node_wise_output_network = MLP(
            in_dim=hidden_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators),
                          hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm,
                          batch_norm_momentum = batch_norm_momentum,
                          out_dim=target_dim,
                          layers=readout_layers)

    def forward(self, graph: dgl.DGLGraph):
        graph.apply_nodes(self.input_node_func)
        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(graph.edata['d'], num_encodings=self.fourier_encodings).squeeze()

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_node_func(self, nodes):
        return {'feat': F.silu(self.input(nodes.data['feat']))}

    def input_edge_func(self, edges):
        return {'d': F.silu(self.edge_input(edges.data['d']))}


class EGCLayer(nn.Module):
    def __init__(self, node_dim, reduce_func, edge_dim, hidden_dim, batch_norm, batch_norm_momentum, dropout, mid_activation):
        super(EGCLayer, self).__init__()
        self.message_network = MLP(
            in_dim=hidden_dim * 2 + edge_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation=mid_activation,
        )
        if reduce_func=='sum':
            self.reduce_func = fn.sum
        elif reduce_func=='mean':
            self.reduce_func = fn.mean
        else:
            raise ValueError('reduce function not supportet: ', reduce_func)
        self.update_network = MLP(
            in_dim=hidden_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )

        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=self.reduce_func(msg='m', out='m_sum'),
                         apply_node_func=self.update_function)

    def message_function(self, edges):
        squared_distance = torch.sum((edges.src['x'] - edges.dst['x']) ** 2, dim=-1)[:, None]
        message_input = torch.cat([edges.src['feat'], edges.dst['feat'], squared_distance], dim=-1)
        message = self.message_network(message_input)
        edge_weight = torch.sigmoid(self.soft_edge_network(message))
        return {'m': message * edge_weight}

    def update_function(self, nodes):
        h = nodes.data['feat']
        input = torch.cat([nodes.data['m_sum'] + nodes.data['feat']], dim=-1)
        h_new = self.update_network(input)
        output = h_new + h
        return {'feat': output}
