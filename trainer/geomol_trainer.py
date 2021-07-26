import os
from itertools import chain
from typing import Dict, Callable

import dgl
import torch

from trainer.trainer import Trainer


class GeomolTrainer(Trainer):
    def __init__(self, **kwargs):
        super(GeomolTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        info2d, info3d, *snorm_n = tuple(batch)
        loss = self.model(*info2d, *snorm_n)  # foward the rest of the batch to the model
        return loss

