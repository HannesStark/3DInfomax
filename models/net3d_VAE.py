from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from commons.mol_encoder import AtomEncoder
from commons.utils import fourier_encode_dist
from models.base_layers import MLP
from models.net3d import Net3DLayer


class Net3DVAE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, readout_aggregators: List[str], batch_norm=False,
                 node_wise_output_layers=2, batch_norm_momentum=0.1, reduce_func='sum',
                 dropout=0.0, propagation_depth: int = 4,
                 fourier_encodings=0, activation: str = 'SiLU', update_net_layers=2, message_net_layers=2, use_node_features=False, **kwargs):
        super(Net3DVAE, self).__init__()
        self.fourier_encodings = fourier_encodings
        edge_in_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1
        self.edge_input = MLP(in_dim=edge_in_dim, hidden_size=hidden_dim, out_dim=hidden_dim, mid_batch_norm=batch_norm,
                              last_batch_norm=batch_norm, batch_norm_momentum=batch_norm_momentum, layers=1,
                              mid_activation=activation, dropout=dropout, last_activation=activation,
                              )

        self.use_node_features = use_node_features
        if self.use_node_features:
            self.atom_encoder = AtomEncoder(hidden_dim)
        else:
            self.node_embedding = nn.Parameter(torch.empty((hidden_dim,)))
            nn.init.normal_(self.node_embedding)

        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(
                Net3DLayer(edge_dim=hidden_dim, hidden_dim=hidden_dim, batch_norm=batch_norm,
                           batch_norm_momentum=batch_norm_momentum, dropout=dropout, mid_activation=activation,
                           reduce_func=reduce_func, message_net_layers=message_net_layers,
                           update_net_layers=update_net_layers))

        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MLP(in_dim=hidden_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                                mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                                batch_norm_momentum=batch_norm_momentum, layers=node_wise_output_layers,
                                                mid_activation=activation, dropout=dropout, last_activation='None',
                                                )

        self.readout_aggregators = readout_aggregators


    def forward(self, graph: dgl.DGLGraph):
        if self.use_node_features:
            graph.ndata['feat'] = self.atom_encoder(graph.ndata['feat'])
        else:
            graph.ndata['feat'] = self.node_embedding[None, :].expand(graph.number_of_nodes(), -1)

        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(graph.edata['d'], num_encodings=self.fourier_encodings)
        graph.apply_edges(self.input_edge_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        if self.node_wise_output_layers > 0:
            graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return readout

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_edge_func(self, edges):
        return {'d': F.silu(self.edge_input(edges.data['d']))}

