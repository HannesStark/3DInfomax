from typing import Dict

import dgl
import torch

from torch import nn

from models.base_layers import MLP
from models.pna import PNAGNN


class DistancePredictor(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self, hidden_dim, target_dim, projection_dim  = 3, distance_net=False, projection_layers=1, **kwargs):
        super(DistancePredictor, self).__init__()
        self.gnn = PNAGNN(hidden_dim=hidden_dim, **kwargs)

        if projection_dim > 0:
            self.node_projection_net = MLP(in_dim=hidden_dim, hidden_size=32,
                                    mid_batch_norm=True, out_dim=projection_dim,
                                    layers=projection_layers)
        else:
            self.node_projection_net = None
        if distance_net:
            self.distance_net = MLP(in_dim=hidden_dim * 2, hidden_size=projection_dim,
                                mid_batch_norm=True, out_dim=target_dim,
                                layers=projection_layers)
        else:
            self.distance_net = None

    def forward(self, mol_graph: dgl.DGLGraph, pairwise_indices: torch.Tensor):
        # get embeddings
        self.gnn(mol_graph)

        # apply down projection to embeddings if we are not using a distance net and projection_dim > 0
        if self.node_projection_net and not self.distance_net:
            mol_graph.apply_nodes(self.node_projection)

        # put the embeddings h from the same graph in the batched graph into pairs for the distance net to predict the pairwise distances
        h = mol_graph.ndata['f']
        src_h = torch.index_select(h, dim=0, index=pairwise_indices[0])
        dst_h = torch.index_select(h, dim=0, index=pairwise_indices[1])

        # for debugging:
        # x = mol_graph.ndata['x']
        # src_x = torch.index_select(x, dim=0, index=pairwise_indices[0])
        # dst_x = torch.index_select(x, dim=0, index=pairwise_indices[1])
        # ic(torch.norm(src_x-dst_x, dim=-1))

        if self.distance_net:
            src_dst_h = torch.cat([src_h, dst_h], dim=1)
            distances = self.distance_net(src_dst_h)
        else:
            distances = torch.norm(src_h-dst_h, dim=-1).unsqueeze(-1)

        return distances

    def node_projection(self, nodes):
        return {'f': self.node_projection_net(nodes.data['f'])}

    def distance_function(self, edges) -> Dict[str, torch.Tensor]:
        src_dst = torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
        return {"distances": self.distance_net(src_dst)}
