import math
from typing import List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from commons.utils import fourier_encode_dist
from models.base_layers import MLP


class DistanceEncoder(nn.Module):
    def __init__(self, hidden_dim, target_dim,
                 batch_norm=False,
                 readout_batchnorm=True, batch_norm_momentum=0.1,
                 dropout=0.0, readout_layers: int = 2, readout_hidden_dim=None,
                 fourier_encodings=0,
                 activation: str = 'SiLU', **kwargs):
        super(DistanceEncoder, self).__init__()
        self.fourier_encodings = fourier_encodings

        input_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1
        self.input_net = MLP(
            in_dim=input_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=1,
            mid_activation=activation,
            dropout=dropout,
            last_activation=activation,
        )

        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.output = MLP(in_dim=hidden_dim * 4, hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm,
                          batch_norm_momentum=batch_norm_momentum,
                          out_dim=target_dim,
                          layers=readout_layers)

    def forward(self, distances, mask):
        if self.fourier_encodings > 0:
            distances = fourier_encode_dist(distances, num_encodings=self.fourier_encodings)
        w = self.input_net(distances)

        mask = mask[..., None].repeat(1, 1, w.shape[-1])


        w_max, _ = torch.max(w - mask * 10e10, dim=1)
        w_min, _ = torch.min(w + mask * 10e10, dim=1)
        w_mean = torch.sum(w * ~mask, dim=1) / (~mask*1).sum(dim=1)
        w_sum = torch.sum(w * ~mask, dim=1)

        readout = torch.cat([w_max, w_sum, w_mean, w_min], dim=-1)
        return self.output(readout)
