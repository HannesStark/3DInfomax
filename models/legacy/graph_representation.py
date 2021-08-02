from typing import Dict

import dgl
import torch

from torch import nn

from models.base_layers import MLP
from models.pna import PNAGNN


class GraphRepresentation(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self, hidden_dim, target_dim, **kwargs):
        super(GraphRepresentation, self).__init__()
        self.gnn = PNAGNN(hidden_dim=hidden_dim, **kwargs)
        self.distance_net = MLP(in_dim=hidden_dim * 2, hidden_size=32,
                                mid_batch_norm=True, out_dim=target_dim,
                                layers=2)

    def forward(self, dgl_graph: dgl.DGLGraph, pairwise_indices: torch.Tensor):
        # get embeddings
        self.gnn(dgl_graph)

        # put the embeddings h from the same graph in the batched graph into pairs for the distance net to predict the pairwise distances
        h = dgl_graph.ndata['feat']
        src_h = torch.index_select(h, dim=0, index=pairwise_indices[0])
        dst_h = torch.index_select(h, dim=0, index=pairwise_indices[1])
        src_dst_h = torch.cat([src_h, dst_h], dim=1)

        # for debugging:
        # x = dgl_graph.ndata['x']
        # src_x = torch.index_select(x, dim=0, index=pairwise_indices[0])
        # dst_x = torch.index_select(x, dim=0, index=pairwise_indices[1])
        # ic(torch.norm(src_x-dst_x, dim=-1))

        return self.distance_net(src_dst_h)

    def distance_function(self, edges) -> Dict[str, torch.Tensor]:
        src_dst = torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
        return {"distances": self.distance_net(src_dst)}
