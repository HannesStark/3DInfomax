from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from models.base_layers import MLP


class GCN(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim,
                 target_dim,
                 readout_aggregators: List[str],
                 readout_batchnorm: bool = True,
                 readout_hidden_dim=None,
                 readout_layers: int = 2,
                 propagation_depth: int = 5,
                 **kwargs):
        super(GCN, self).__init__()
        self.node_gnn = GCNGNN(node_dim=node_dim,
                          hidden_dim=hidden_dim,
                          propagation_depth=propagation_depth,
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


class GCNGNN(nn.Module):
    def __init__(self,
                 node_dim,
                 hidden_dim,
                 propagation_depth: int = 5):
        super(GCNGNN, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(GraphConv(node_dim, hidden_dim))
        for _ in range(propagation_depth - 1):
            self.convolutions.append(GraphConv(hidden_dim, hidden_dim))

    def forward(self, graph):
        h = graph.ndata['feat']
        for convolution in self.convolutions:
            h = F.relu(convolution(graph, h))
        graph.ndata['feat'] = h

