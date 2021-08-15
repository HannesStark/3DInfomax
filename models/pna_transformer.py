from typing import Dict, List, Union, Callable

import torch
import numpy as np
from functools import partial

from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from commons.mol_encoder import AtomEncoder, BondEncoder
from models.base_layers import MLP
from models.pna import PNALayer
from models.pna_original import PNASimpleLayer

EPS = 1e-5


class PNATransformer(nn.Module):

    def __init__(self, hidden_dim, target_dim, dropout, nhead, dim_feedforward, aggregators: List[str],
                 scalers: List[str], readout_batchnorm: bool = True, readout_hidden_dim=None, readout_layers: int = 2,
                 batch_norm_momentum=0.1, pos_enc_dim=16, residual: bool = True, pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu", last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False, last_batch_norm: bool = False, propagation_depth: int = 5,
                 posttrans_layers: int = 1, pretrans_layers: int = 1, simple=False, **kwargs):
        super(PNATransformer, self).__init__()
        self.node_gnn = PNATransformerGNN( hidden_dim=hidden_dim, dim_feedforward=dim_feedforward,
                                          nhead=nhead, dropout=dropout, activation=activation, aggregators=aggregators,
                                          scalers=scalers, pairwise_distances=pairwise_distances, residual=residual,
                                          last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                                          last_batch_norm=last_batch_norm, avg_d={"log": 1.0},
                                          posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                                          batch_norm_momentum=batch_norm_momentum,simple=simple)

        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.output = MLP(in_dim=hidden_dim, hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, dgl_graph, h, pos_enc, mask):
        batch_size, max_num_atoms, _ = h.size()

        # node gnn adds a virtual node for readout
        h = self.node_gnn(dgl_graph, h, pos_enc, mask)  # [batch_size, max_num_atoms + 1, hidden_dim]

        # the first node is the one that was added for readout
        return self.output(h[:, 0, :])


class PNATransformerGNN(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward, aggregators: List[str], scalers: List[str],
                 nhead: int = 4, pos_enc_dim=16, residual: bool = True, pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu", last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False, last_batch_norm: bool = False, batch_norm_momentum=0.1,
                 propagation_depth: int = 5, dropout: float = 0.0, posttrans_layers: int = 1, pretrans_layers: int = 1, simple=False,
                 **kwargs):
        super(PNATransformerGNN, self).__init__()

        self.mp_layers = nn.ModuleList()
        self.pos_enc_mlp = nn.Linear(2, pos_enc_dim)
        self.v_node = nn.Parameter(torch.empty((hidden_dim,)))
        nn.init.normal_(self.v_node)


        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNATransformerLayer(hidden_dim=hidden_dim, in_dim_edges=hidden_dim,
                                    dim_feedforward=dim_feedforward, nhead=nhead, batch_first=True,
                                    aggregators=aggregators, scalers=scalers, pairwise_distances=pairwise_distances,
                                    residual=residual, dropout=dropout, activation=activation,
                                    last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                                    last_batch_norm=last_batch_norm, avg_d={"log": 1.0},
                                    posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                                    batch_norm_momentum=batch_norm_momentum, simple=simple)
            )

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim - pos_enc_dim)
        self.graph_atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, graph, h, pos_enc, mask):
        batch_size, max_num_atoms, _ = h.size()
        graph.ndata['feat'] = self.graph_atom_encoder(graph.ndata['feat'])
        graph.edata['feat'] = self.bond_encoder(graph.edata['feat'])

        h = self.atom_encoder(h.view(-1, h.shape[-1]))  # [batch_size, max_num_atoms * (hidden_dim - pos_enc_dim)]
        h = h.view(batch_size, max_num_atoms, -1)  # [batch_size, max_num_atoms, hidden_dim - pos_enc_dim]
        pos_enc = self.pos_enc_mlp(pos_enc)  # [batch_size, max_num_atoms, num_eigvec, pos_enc_dim]
        pos_enc = pos_enc.nansum(dim=2)  # [batch_size, max_num_atoms, pos_enc_dim]
        h = torch.cat([h, pos_enc], dim=-1)  # [batch_size, max_num_atoms, hidden_dim]

        # add virtual node for readout
        h = torch.cat([self.v_node[None, None, :].expand(batch_size, -1, -1), h],
                      dim=1)  # [batch_size, max_num_atoms + 1, hidden_dim]

        mask_include_vnode = torch.cat([torch.tensor(False, device=h.device).unsqueeze(0).expand(batch_size, -1), mask],
                         dim=1)  # [batch_size, max_num_atoms + 1]

        mask_exclude_vnode = torch.cat([torch.tensor(True, device=h.device).unsqueeze(0).expand(batch_size, -1), mask],
                         dim=1)  # [batch_size, max_num_atoms + 1]

        h_in = h

        for mp_layer in self.mp_layers:
            h = mp_layer(graph, h, mask_include_vnode, mask_exclude_vnode)
        return h


class PNATransformerLayer(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward, aggregators: List[str], scalers: List[str],
                 nhead: int = 4,
                 residual: bool = True, pairwise_distances: bool = False, activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none", mid_batch_norm: bool = False,
                 last_batch_norm: bool = False, batch_norm_momentum=0.1,
                 dropout: float = 0.0, posttrans_layers: int = 1, pretrans_layers: int = 1, combine_mlp_batchnorm_last = False, simple=False, **kwargs):
        super(PNATransformerLayer, self).__init__()

        self.transformer_layer = TransformerEncoderLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward,
                                                         nhead=nhead, batch_first=True, dropout=dropout,
                                                         activation=activation)
        self.simple= simple
        if simple:
            self.pna_layer = PNASimpleLayer(in_dim=hidden_dim, out_dim=hidden_dim,
                                            aggregators=aggregators, scalers=scalers,
                                            residual=residual, dropout=dropout, mid_batch_norm=mid_batch_norm,
                                            last_batch_norm=last_batch_norm, avg_d=1.0,
                                            posttrans_layers=posttrans_layers)
        else:
            self.pna_layer = PNALayer(in_dim=hidden_dim, out_dim=hidden_dim, in_dim_edges=hidden_dim,
                                  aggregators=aggregators, scalers=scalers, pairwise_distances=pairwise_distances,
                                  residual=residual, dropout=dropout, activation=activation,
                                  last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                                  last_batch_norm=last_batch_norm, avg_d={"log": 1.0},
                                  posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                                  batch_norm_momentum=batch_norm_momentum)
        self.combine_mlp = MLP(in_dim=2*hidden_dim, hidden_size=hidden_dim,
                          last_batch_norm=combine_mlp_batchnorm_last, out_dim=hidden_dim,
                          layers=1, batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph, h, mask_include_vnode, mask_exclude_vnode):
        # shape of masks: [batch_size, max_num_atoms + 1]
        batch_size, n_atoms_plus_one, hidden_dim = h.size()

        if self.simple:
            feat = self.pna_layer(graph, graph.ndata['feat'])
            graph.ndata['feat'] = feat
        else:
            self.pna_layer(graph)

        h_graph = graph.ndata['feat'] # [n_nodes, hidden_dim]
        n_atoms, hidden_dim = h_graph.size()

        h_graph_padded = h.clone() # [batch_size, max_num_atoms + 1, hidden_dim]
        h_graph_padded = h_graph_padded.view(-1, hidden_dim) # [batch_size*(max_num_atoms + 1), hidden_dim]
        expanded_mask = mask_exclude_vnode.view(-1).unsqueeze(1).expand(-1, hidden_dim) # [batch_size*(max_num_atoms + 1), hidden_dim]
        h_graph_padded[~expanded_mask] = h_graph.view(-1)

        h_transformer = self.transformer_layer(h, src_key_padding_mask=mask_include_vnode) # [batch_size, max_num_atoms + 1, hidden_dim]

        h = torch.cat([h_graph_padded, h_transformer.view(-1, hidden_dim)], dim=1)   # [batch_size*(max_num_atoms + 1), 2*hidden_dim]
        h = self.combine_mlp(h) # [batch_size*(max_num_atoms + 1), hidden_dim]
        h_graph = h[~expanded_mask]  # [n_atoms*hidden_dim]
        graph.ndata['feat'] = h_graph.view(n_atoms, hidden_dim)
        h = h.view(batch_size, n_atoms_plus_one, hidden_dim) # [batch_size, max_num_atoms + 1, hidden_dim]
        return h
