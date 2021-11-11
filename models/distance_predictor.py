from typing import Dict

import dgl
import torch

from torch import nn
from torch.nn import TransformerEncoderLayer

from models.base_layers import MLP
from models.pna import PNAGNN
import torch.nn.functional as F


class DistancePredictor(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self,target_dim, pna_args, projection_dim=3, distance_net=False, projection_layers=1, transformer_layer=True, nhead=16, dim_feedforward=256, activation='relu', **kwargs):
        super(DistancePredictor, self).__init__()
        hidden_dim = pna_args['hidden_dim']
        dropout = pna_args['dropout']
        self.node_gnn = PNAGNN( **pna_args)
        self.transformer_layer = transformer_layer
        if transformer_layer:
            self.transformer_layer = TransformerEncoderLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward,
                                                         nhead=nhead, batch_first=True, dropout=dropout,
                                                         activation=activation)
        if projection_dim > 0:
            self.node_projection_net = MLP(in_dim=hidden_dim, hidden_size=32, mid_batch_norm=True,
                                           out_dim=projection_dim, layers=projection_layers)
        else:
            self.node_projection_net = None
        if distance_net:
            self.distance_net = MLP(in_dim=hidden_dim * 2, hidden_size=projection_dim, mid_batch_norm=True,
                                    out_dim=target_dim, layers=projection_layers)
        else:
            self.distance_net = None

    def forward(self, graph: dgl.DGLGraph, pairwise_indices: torch.Tensor, mask):
        batch_size, max_num_atoms = mask.size()
        # get embeddings
        self.node_gnn(graph)
        n_atoms, hidden_dim = graph.ndata['feat'].size()

        # apply down projection to embeddings if we are not using a distance net and projection_dim > 0
        if self.node_projection_net and not self.distance_net:
            graph.apply_nodes(self.node_projection)

        if self.transformer_layer:
            # put the embeddings h from the same graph in the batched graph into pairs for the distance net to predict the pairwise distances
            feat = graph.ndata['feat'] # [n_atoms, hidden_dim]

            expanded_mask = mask.view(-1).unsqueeze(1).expand(-1,hidden_dim)  # [batch_size*(max_num_atoms), hidden_dim]
            transformer_feat = torch.zeros_like(expanded_mask, device=mask.device, dtype=torch.float) # [batch_size*(max_num_atoms), hidden_dim]
            transformer_feat[~expanded_mask] = feat.view(-1)
            transformer_feat = self.transformer_layer(transformer_feat.view(batch_size,max_num_atoms, hidden_dim),src_key_padding_mask=mask)  # [batch_size, max_num_atoms, hidden_dim]
            feat = transformer_feat.view(batch_size* max_num_atoms, hidden_dim)[~expanded_mask] # [n_atoms*hidden_dim]
            graph.ndata['feat'] = feat.view(n_atoms, hidden_dim)

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

    def node_projection(self, nodes):
        return {'feat': self.node_projection_net(nodes.data['feat'])}

    def distance_function(self, edges) -> Dict[str, torch.Tensor]:
        src_dst = torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
        return {"distances": self.distance_net(src_dst)}
