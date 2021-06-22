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


class Transformer(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self,
                 node_dim,
                 hidden_dim,
                 target_dim,
                 dropout,
                 nhead,
                 dim_feedforward,
                 readout_aggregators: List[str],
                 readout_batchnorm: bool = True,
                 readout_hidden_dim=None,
                 activation: str = 'relu',
                 readout_layers: int = 2,
                 batch_norm_momentum=0.1,
                 **kwargs):
        super(Transformer, self).__init__()
        self.node_gnn = TransformerGNN(node_dim=node_dim, hidden_dim=hidden_dim, dim_feedforward=dim_feedforward,
                                       nhead=nhead, dropout=dropout, activation=activation)
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, nodes):
        emb = self.node_gnn(nodes)
        return self.output(emb)


class TransformerGNN(nn.Module):
    def __init__(self,
                 node_dim,
                 hidden_dim,
                 dim_feedforward,
                 nhead: int = 4,
                 activation: Union[Callable, str] = "relu",
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 **kwargs):
        super(TransformerGNN, self).__init__()

        self.mp_layers = nn.ModuleList()

        for _ in range(propagation_depth):
            self.mp_layers.append(
                TransformerEncoderLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, nhead=nhead,
                                        dropout=dropout, activation=activation))
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, nodes):




