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


class Net3DDistancePredictor(nn.Module):
    def __init__(self, hidden_dim, readout_aggregators: List[str], batch_norm=False,
                 node_wise_encoder_layers=0, node_wise_output_layers = 0, batch_norm_momentum=0.1, reduce_func='sum',
                 dropout=0.0, propagation_depth: int = 4, decoder_depth: int = 0, projection_dim=3, distance_net=True, projection_layers=1,
                 fourier_encodings=0, activation: str = 'SiLU', update_net_layers=2, message_net_layers=2, use_node_features=False, **kwargs):
        super(Net3DDistancePredictor, self).__init__()
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
        self.decoder_layers = nn.ModuleList()
        for _ in range(decoder_depth):
            self.decoder_layers.append(
                Net3DLayer(edge_dim=hidden_dim, hidden_dim=hidden_dim, batch_norm=batch_norm,
                           batch_norm_momentum=batch_norm_momentum, dropout=dropout, mid_activation=activation,
                           reduce_func=reduce_func, message_net_layers=message_net_layers,
                           update_net_layers=update_net_layers))

        self.node_wise_encoder_layers = node_wise_encoder_layers
        if self.node_wise_encoder_layers > 0:
            self.node_wise_output_network = MLP(in_dim=hidden_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                                mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                                batch_norm_momentum=batch_norm_momentum, layers=node_wise_encoder_layers,
                                                mid_activation=activation, dropout=dropout, last_activation='None',
                                                )
        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MLP(in_dim=hidden_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                                mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                                batch_norm_momentum=batch_norm_momentum,
                                                layers=node_wise_output_layers,
                                                mid_activation=activation, dropout=dropout, last_activation='None',
                                                )

        self.readout_aggregators = readout_aggregators
        if projection_dim > 0 and distance_net==False:
            self.node_projection_net = MLP(in_dim=hidden_dim, hidden_size=32, mid_batch_norm=True,
                                           out_dim=projection_dim, layers=projection_layers)
        else:
            self.node_projection_net = None
        if distance_net:
            self.distance_net = MLP(in_dim=hidden_dim * 2, hidden_size=projection_dim, mid_batch_norm=True,
                                    out_dim=1, layers=projection_layers)
        else:
            self.distance_net = None


    def forward(self, graph: dgl.DGLGraph, pairwise_indices: torch.Tensor, mask):
        if self.use_node_features:
            graph.ndata['feat'] = self.atom_encoder(graph.ndata['feat'])
        else:
            graph.ndata['feat'] = self.node_embedding[None, :].expand(graph.number_of_nodes(), -1)

        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(graph.edata['d'], num_encodings=self.fourier_encodings)
        graph.apply_edges(self.input_edge_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        if self.node_wise_encoder_layers > 0:
            graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        latent_vector = torch.cat(readouts_to_cat, dim=-1)

        for mp_layer in self.decoder_layers:
            mp_layer(graph)
        n_atoms, hidden_dim = graph.ndata['feat'].size()

        # apply down projection to embeddings if we are not using a distance net and projection_dim > 0
        if self.node_projection_net and not self.distance_net:
            graph.apply_nodes(self.node_projection)


        h = graph.ndata['feat']
        src_h = torch.index_select(h, dim=0, index=pairwise_indices[0])
        dst_h = torch.index_select(h, dim=0, index=pairwise_indices[1])

        # for debugging:
        #x = graph.ndata['x']
        #src_x = torch.index_select(x, dim=0, index=pairwise_indices[0])
        #dst_x = torch.index_select(x, dim=0, index=pairwise_indices[1])

        if self.distance_net:
            src_dst_h = torch.cat([src_h, dst_h], dim=1)
            dst_src_h = torch.cat([dst_h, src_h], dim=1)
            distances = F.softplus(self.distance_net(src_dst_h) + self.distance_net(dst_src_h))
        else:
            distances = torch.norm(src_h - dst_h, dim=-1).unsqueeze(-1)

        # for debugging
        #distances = torch.norm(src_x - dst_x, p=2, dim=-1).unsqueeze(-1)
        return distances


    def node_projection(self, nodes):
        return {'feat': self.node_projection_net(nodes.data['feat'])}

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_edge_func(self, edges):
        return {'d': F.silu(self.edge_input(edges.data['d']))}

