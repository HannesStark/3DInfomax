from typing import Callable, List, Union

import torch
import torch.nn as nn

from models.base_layers import MLP


class BasicCritic(nn.Module):

    def __init__(self,
                 metric_dim,
                 repeats,
                 dropout=0.8,
                 mid_batch_norm=True,
                 last_batch_norm=True,
                 **kwargs):
        super(BasicCritic, self).__init__()
        self.repeats = repeats
        self.dropout = dropout
        self.criticise = MLP(in_dim=metric_dim * repeats, hidden_size=metric_dim * repeats,
                             mid_batch_norm=mid_batch_norm, out_dim=metric_dim * repeats,
                             last_batch_norm=last_batch_norm,
                             dropout=0,
                             layers=2)

    def forward(self, philosophy):
        batch_size, metric_dim = philosophy.size()
        philosophy = philosophy[..., None]  # [batchsize, metric_dim, 1]
        philosophy = philosophy.repeat(1, 1, self.repeats).view((batch_size, -1))
        dropout_mask = torch.rand_like(philosophy, device=philosophy.device) > self.dropout
        philosophy = philosophy * dropout_mask
        criticism = self.criticise(philosophy)

        return criticism.view(batch_size, metric_dim, -1)
