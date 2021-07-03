from typing import Dict, List, Union, Callable

import torch
import numpy as np
from functools import partial

from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from commons.mol_encoder import AtomEncoder, BondEncoder
from models.base_layers import MLP

EPS = 1e-5


class TransformerPlain(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self, node_dim, hidden_dim, target_dim, dropout, nhead, dim_feedforward,
                 readout_batchnorm: bool = True, readout_hidden_dim=None, activation: str = 'relu',
                 readout_layers: int = 2, batch_norm_momentum=0.1, **kwargs):
        super(TransformerPlain, self).__init__()
        self.node_gnn = TransformerGNN(node_dim=node_dim, hidden_dim=hidden_dim, dim_feedforward=dim_feedforward,
                                       nhead=nhead, dropout=dropout, activation=activation)


        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.output = MLP(in_dim=hidden_dim, hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, h, pos_enc, mask):
        batch_size, max_num_atoms, _ = h.size()

        # node gnn adds a virtual node for readout
        h = self.node_gnn(h, pos_enc, mask) # [batch_size, max_num_atoms + 1, hidden_dim]

        # the first node is the one that was added for readout
        return self.output(h[:,0,:])


class TransformerGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim, dim_feedforward, nhead: int = 4, pos_enc_dim=16,
                 activation: Union[Callable, str] = "relu", propagation_depth: int = 5, dropout: float = 0.0, **kwargs):
        super(TransformerGNN, self).__init__()

        self.mp_layers = nn.ModuleList()
        self.pos_enc_mlp = nn.Linear(2, pos_enc_dim)
        self.v_node = nn.Parameter(torch.empty((hidden_dim,)))
        nn.init.normal_(self.v_node)

        for _ in range(propagation_depth):
            self.mp_layers.append(
                TransformerEncoderLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, nhead=nhead,
                                        batch_first=True, dropout=dropout, activation=activation))
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim - pos_enc_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, h, pos_enc, mask):
        batch_size, max_num_atoms, _ = h.size()

        h = self.atom_encoder(h.view(-1, h.shape[-1]))  # [batch_size, max_num_atoms * (hidden_dim - pos_enc_dim)]
        h = h.view(batch_size, max_num_atoms, -1)  # [batch_size, max_num_atoms, hidden_dim - pos_enc_dim]
        pos_enc = self.pos_enc_mlp(pos_enc)  # [batch_size, max_num_atoms, num_eigvec, pos_enc_dim]
        pos_enc = pos_enc.nansum(dim=2)  # [batch_size, max_num_atoms, pos_enc_dim]
        h = torch.cat([h, pos_enc], dim=-1)  # [batch_size, max_num_atoms, hidden_dim]

        # add virtual node for readout
        h = torch.cat([self.v_node[None, None, :].expand(batch_size, -1, -1), h], dim=1)# [batch_size, max_num_atoms + 1, hidden_dim]

        mask = torch.cat([torch.tensor(False, device=h.device).unsqueeze(0).expand(batch_size, -1), mask], dim=1)# [batch_size, max_num_atoms + 1]

        h_in = h

        for mp_layer in self.mp_layers:
            h = mp_layer(h, src_key_padding_mask=mask)
        return h
