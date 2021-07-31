import torch

from commons.utils import move_to_device
from trainer.trainer import Trainer


class GeomolTrainer(Trainer):
    def __init__(self, **kwargs):
        super(GeomolTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch, epoch):
        graphs = tuple(batch)[0]
        loss = self.model(graphs) if epoch > 1 else self.model(graphs,
                                                               ignore_neighbors=True)  # foward the rest of the batch to the model
        return loss

    def process_batch(self, batch, optim, epoch):
        if optim != None:
            self.optim.zero_grad()
        loss = self.forward_pass(batch, epoch)
        if optim != None:  # run backpropagation if an optimizer is provided
            loss.backward()
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim_steps += 1
        return loss.item()

    def predict(self, data_loader, epoch: int, optim: torch.optim.Optimizer = None,
                return_predictions: bool = False):
        total_metrics = {k: 0 for k in
                         ['bond_angle_loss', 'one_hop_loss', 'three_hop_loss', 'torsion_angle_loss', 'two_hop_loss'] + [
                             type(self.loss_func).__name__]}
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            batch = move_to_device(list(batch), self.device)
            loss = self.process_batch(batch, optim, epoch)
            with torch.no_grad():
                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    print('[Epoch %d; Iter %5d/%5d] %s: loss: %.7f' % (
                        epoch, i + 1, len(data_loader), 'train', total_metrics[type(self.loss_func).__name__]))
                    total_metrics = {k: 0 for k in total_metrics.keys()}
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader

                    for key, value in total_metrics.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch:
                    pass
        if optim == None:
            if self.val_per_batch:
                total_metrics = {k: v / len(data_loader) for k, v in total_metrics.items()}
            else:
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(data_loader)
            return total_metrics

    def tensorboard_log(self, metrics, data_split: str, epoch: int, step: int, log_hparam: bool = False):
        metrics['epoch'] = epoch
        for i, param_group in enumerate(self.optim.param_groups):
            metrics[f'lr_param_group_{i}'] = param_group['lr']
        logs = {}
        for key, metric in metrics.items():
            metric_name = f'{key}/{data_split}'
            logs[metric_name] = metric
            self.writer.add_scalar(metric_name, metric, step)
