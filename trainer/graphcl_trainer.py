import torch

from commons.utils import move_to_device
from trainer.trainer import Trainer


class GraphCLTrainer(Trainer):
    def __init__(self, **kwargs):
        super(GraphCLTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        view1, view2 = tuple(batch)
        predictions = self.model(*view1)
        targets = self.model(*view2)
        return self.loss_func(predictions, targets), predictions, targets
