from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from commons.utils import fourier_encode_dist
from models.base_layers import MLP


class DistanceAggregator(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, target_dim, readout_aggregators: List[str],
                 batch_norm=False,
                 node_wise_output_layers=2,
                 readout_batchnorm=True, batch_norm_momentum=0.1, reduce_func='sum',
                 dropout=0.0, readout_layers: int = 2, readout_hidden_dim=None,
                 fourier_encodings=0,
                 activation: str = 'SiLU', **kwargs):
        super(DistanceAggregator, self).__init__()
        self.fourier_encodings = fourier_encodings

        if reduce_func == 'sum':
            self.reduce_func = fn.sum
        elif reduce_func == 'mean':
            self.reduce_func = fn.mean
        else:
            raise ValueError('reduce function not supportet: ', reduce_func)

        edge_in_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1
        self.edge_input = MLP(
            in_dim=edge_in_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=1,
            mid_activation=activation,
            dropout=dropout,
            last_activation=activation,
        )

        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MLP(
                in_dim=hidden_dim,
                hidden_size=hidden_dim,
                out_dim=hidden_dim,
                mid_batch_norm=batch_norm,
                last_batch_norm=batch_norm,
                batch_norm_momentum=batch_norm_momentum,
                layers=node_wise_output_layers,
                mid_activation=activation,
                dropout=dropout,
                last_activation='None',
            )
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm,
                          batch_norm_momentum=batch_norm_momentum,
                          out_dim=target_dim,
                          layers=readout_layers)

    def forward(self, graph: dgl.DGLGraph):
        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(graph.edata['d'], num_encodings=self.fourier_encodings)
        graph.apply_edges(self.input_edge_func)

        graph.update_all(message_func=self.message_function, reduce_func=self.reduce_func(msg='m', out='m_sum'))

        if self.node_wise_output_layers > 0:
            graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)

    def input_node_func(self, nodes):
        return {'feat': F.silu(self.input(nodes.data['feat']))}

    def input_edge_func(self, edges):
        return {'d': F.silu(self.edge_input(edges.data['d']))}

    def message_function(self, edges):
        return {'m': edges.data['d']}

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['m_sum'])}
