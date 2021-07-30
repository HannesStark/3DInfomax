import os
from itertools import chain
from typing import Dict, Callable

import dgl
import torch

from commons.utils import move_to_device
from trainer.trainer import Trainer


class GeomolTrainer(Trainer):
    def __init__(self, **kwargs):
        super(GeomolTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        graphs = tuple(batch)[0]
        loss = self.model(graphs)  # foward the rest of the batch to the model
        return loss

    def process_batch(self, batch, optim):
        loss = self.forward_pass(batch)
        if optim != None:  # run backpropagation if an optimizer is provided
            loss.backward()
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return loss

    def predict(self, data_loader, epoch: int = 0, optim: torch.optim.Optimizer = None,
                return_predictions: bool = False):
        total_metrics = {k: 0 for k in list(self.metrics.keys()) + [type(self.loss_func).__name__]}
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            batch = move_to_device(list(batch), self.device)
            loss = self.process_batch(batch, optim)
            with torch.no_grad():
                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    metrics = {}
                    metrics[type(self.loss_func).__name__] = loss.item()
                    self.tensorboard_log(metrics, data_split='train', step=self.optim_steps, epoch=epoch)
                    print('[Epoch %d; Iter %5d/%5d] %s: loss: %.7f' % (
                        epoch, i + 1, len(data_loader), 'train', loss.item()))
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics_results = {}
                    metrics_results[type(self.loss_func).__name__] = loss.item()
                    for key, value in metrics_results.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch:
                    epoch_loss += loss.item()

        if optim == None:
            if self.val_per_batch:
                total_metrics = {k: v / len(data_loader) for k, v in total_metrics.items()}
            else:
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(data_loader)
            return total_metrics