from typing import List, Union, Callable

import dgl
import torch


from torch import nn
from torch.nn import TransformerEncoderLayer

from models.base_layers import MLP
from models.pna import PNAGNN


class PNASelfAttentionReadout(nn.Module):
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
                 batch_norm_momentum=0.1,
                 nhead= 4,
                 dim_feedforward=256,
                 **kwargs):
        super(PNASelfAttentionReadout, self).__init__()
        self.node_gnn = PNAGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, aggregators=aggregators,
                               scalers=scalers, residual=residual, pairwise_distances=pairwise_distances,
                               activation=activation, last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm, propagation_depth=propagation_depth, dropout=dropout,
                               posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                               batch_norm_momentum=batch_norm_momentum
                               )

        self.transformer_layer = TransformerEncoderLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward,
                                                         nhead=nhead, batch_first=True, dropout=dropout,
                                                         activation=activation)
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph: dgl.DGLGraph, mask):
        batch_size, max_num_atoms = mask.size()
        self.node_gnn(graph)
        feat = graph.ndata['feat']
        n_atoms, hidden_dim = feat.size()

        expanded_mask = mask.view(-1).unsqueeze(1).expand(-1, hidden_dim)  # [batch_size*(max_num_atoms), hidden_dim]
        transformer_feat = torch.zeros_like(expanded_mask, device=mask.device,
                                            dtype=torch.float)  # [batch_size*(max_num_atoms), hidden_dim]
        transformer_feat[~expanded_mask] = feat.view(-1)
        transformer_feat = self.transformer_layer(transformer_feat.view(batch_size, max_num_atoms, hidden_dim),
                                                  src_key_padding_mask=mask)  # [batch_size, max_num_atoms, hidden_dim]
        feat = transformer_feat.view(batch_size * max_num_atoms, hidden_dim)[~expanded_mask]  # [n_atoms*hidden_dim]
        graph.ndata['feat'] = feat.view(n_atoms, hidden_dim)
        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)

