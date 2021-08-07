import torch

from commons.utils import move_to_device
from trainer.trainer import Trainer


class OptimalTransportTrainer(Trainer):
    def __init__(self, **kwargs):
        super(OptimalTransportTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch, epoch):
        data = tuple(batch)
        loss = self.model(*data) if epoch > self.args.num_epochs_local_only else self.model(*data,
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
                    total_metrics['one_hop_loss'] += self.model.one_hop_loss_write.item()
                    total_metrics['two_hop_loss'] += self.model.two_hop_loss_write.item()
                    total_metrics['bond_angle_loss'] += self.model.angle_loss_write.item()
                    total_metrics['torsion_angle_loss'] += self.model.dihedral_loss_write.item()
                    total_metrics['three_hop_loss'] += self.model.three_hop_loss_write.item()
                    total_metrics[type(self.loss_func).__name__] += loss
                    self.tensorboard_log(total_metrics, data_split='train', step=self.optim_steps, epoch=epoch)
                    print('[Epoch %d; Iter %5d/%5d] %s: loss: %.7f' % (
                        epoch, i + 1, len(data_loader), 'train', total_metrics[type(self.loss_func).__name__]))
                    total_metrics = {k: 0 for k in total_metrics.keys()}
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    total_metrics['one_hop_loss'] += self.model.one_hop_loss_write.item()
                    total_metrics['two_hop_loss'] += self.model.two_hop_loss_write.item()
                    total_metrics['bond_angle_loss'] += self.model.angle_loss_write.item()
                    total_metrics['torsion_angle_loss'] += self.model.dihedral_loss_write.item()
                    total_metrics['three_hop_loss'] += self.model.three_hop_loss_write.item()
                    total_metrics[type(self.loss_func).__name__] += loss

                if optim == None and not self.val_per_batch:
                    epoch_loss += loss.item()

        if optim == None:
            if self.val_per_batch:
                total_metrics = {k: v / len(data_loader) for k, v in total_metrics.items()}
            else:
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(data_loader)
            return total_metrics
