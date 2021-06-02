from typing import Callable, List, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from models.base_layers import MLP


class BasicCritic(nn.Module):
    """
    Message Passing Neural Network
    """

    def __init__(self,
                 metric_dim,
                 repeats,
                 dropout=0.8,
                 mid_batch_norm=True,
                 last_batch_norm=True,
                 **kwargs):
        super(BasicCritic, self).__init__()
        self.repeats = repeats
        self.criticise = MLP(in_dim=metric_dim * repeats, hidden_size=metric_dim * repeats,
                             mid_batch_norm=mid_batch_norm, out_dim=metric_dim * repeats,
                             last_batch_norm=last_batch_norm,
                             dropout=dropout,
                             layers=2)

    def forward(self, philosophy):
        batch_size, metric_dim = philosophy.size()
        philosophy = philosophy[..., None]  # [batchsize, metric_dim, 1]
        philosophy = philosophy.repeat(1, 1, self.repeats).view((batch_size, -1))
        criticism = self.criticise(philosophy)

        return criticism.view(batch_size,metric_dim,-1)
