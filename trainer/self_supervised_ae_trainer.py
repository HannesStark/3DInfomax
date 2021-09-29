from typing import Union, Tuple, Dict

import torch
from torch.utils.data import DataLoader

from commons.utils import move_to_device
from trainer.self_supervised_trainer import SelfSupervisedTrainer


class SelfSupervisedAETrainer(SelfSupervisedTrainer):
    def __init__(self, **kwargs):
        super(SelfSupervisedAETrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        graph, info3d, distances = tuple(batch)
        view2d = self.model(*graph)  # foward the rest of the batch to the model
        view3d, distance_preds = self.model3d(*info3d)
        loss_contrastive, loss_reconstruction = self.loss_func(view2d, view3d, distance_preds, distances)
        return loss_contrastive, loss_reconstruction, view2d, view3d

    def process_batch(self, batch, optim):
        loss_contrastive,loss_reconstruction, predictions, targets = self.forward_pass(batch)
        loss = loss_contrastive + loss_reconstruction
        if optim != None:  # run backpropagation if an optimizer is provided
            loss.backward()
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return loss_contrastive, loss_reconstruction, predictions.detach(), targets.detach()

    def predict(self, data_loader: DataLoader, epoch: int, optim: torch.optim.Optimizer = None,
                return_predictions: bool = False) -> Union[
        Dict, Tuple[float, Union[torch.Tensor, None], Union[torch.Tensor, None]]]:
        total_metrics = {k: 0 for k in
                         list(self.metrics.keys()) + [type(self.loss_func).__name__, 'mean_pred', 'std_pred',
                                                      'mean_targets', 'std_targets', 'contrastive_loss', 'reconstruction_loss']}
        epoch_targets = torch.tensor([]).to(self.device)
        epoch_predictions = torch.tensor([]).to(self.device)
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            batch = move_to_device(list(batch), self.device)
            loss_contrastive,loss_reconstruction, predictions, targets = self.process_batch(batch, optim)
            with torch.no_grad():
                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    metrics = self.evaluate_metrics(predictions, targets)
                    metrics[type(self.loss_func).__name__] = loss_contrastive.item() + loss_reconstruction.item()
                    metrics['contrastive_loss'] = loss_contrastive.item()
                    metrics['reconstruction_loss'] = loss_reconstruction.item()
                    self.run_tensorboard_functions(predictions, targets, step=self.optim_steps, data_split='train')
                    self.tensorboard_log(metrics, data_split='train', step=self.optim_steps, epoch=epoch)
                    print('[Epoch %d; Iter %5d/%5d] %s: loss: %.7f' % (
                        epoch, i + 1, len(data_loader), 'train', loss_contrastive.item() +loss_reconstruction.item()))
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics_results = self.evaluate_metrics(predictions, targets, val=True)
                    metrics_results[type(self.loss_func).__name__] = loss_contrastive.item() + loss_reconstruction.item()
                    metrics_results['contrastive_loss'] = loss_contrastive.item()
                    metrics_results['reconstruction_loss'] = loss_reconstruction.item()
                    if i ==0 and epoch in self.args.models_to_save:
                        self.run_tensorboard_functions(predictions, targets, step=self.optim_steps, data_split='val')
                    for key, value in metrics_results.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch:
                    epoch_loss += loss_contrastive.item() + loss_reconstruction.item()
                    epoch_targets = torch.cat((targets, epoch_targets), 0)
                    epoch_predictions = torch.cat((predictions, epoch_predictions), 0)

        if optim == None:
            if self.val_per_batch:
                total_metrics = {k: v / len(data_loader) for k, v in total_metrics.items()}
            else:
                total_metrics = self.evaluate_metrics(epoch_predictions, epoch_targets, val=True)
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(data_loader)
            return total_metrics