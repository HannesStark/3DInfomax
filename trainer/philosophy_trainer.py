from typing import Tuple, Union, Dict

import torch
from torch.utils.data import DataLoader

from trainer.lr_schedulers import WarmUpWrapper

from trainer.self_supervised_trainer import SelfSupervisedTrainer


class PhilosophyTrainer(SelfSupervisedTrainer):
    # TODO: implement loading of checkpoints to continue training. The current problem is that only one optimizer and one scheduler will be saved. We need to overwrite save model and move the scheduler and optim loading from statedict into the initialize_optim and initialize_scheduler functions instead of having them in the constructor of Trainer
    def __init__(self, critic, critic_loss, device: torch.device, **kwargs):
        self.critic = critic.to(device)
        super(PhilosophyTrainer, self).__init__(device=device, **kwargs)
        self.critic_loss_func = critic_loss

    def forward_pass(self, batch):
        graph, info3d, *targets = tuple(batch)
        view2d = self.model(graph)
        view3d = self.model3d(info3d)

        reconstruction = self.critic(view3d)

        critic_loss = self.critic_loss_func(view3d, reconstruction)
        peasant_loss = self.loss_func(view2d, view3d, nodes_per_graph=graph.batch_num_nodes())
        philosopher_loss = peasant_loss - critic_loss
        return peasant_loss, philosopher_loss, critic_loss, view2d, view3d

    def process_batch(self, batch, optim):
        peasant_loss, philosopher_loss, critic_loss, view2d, view3d = self.forward_pass(batch)
        if optim != None:  # run backpropagation if an optimizer is provided
            peasant_loss.backward(inputs=list(self.model.parameters()), retain_graph=True)
            self.optim.step()
            philosopher_loss.backward(inputs=list(self.model3d.parameters()), retain_graph=True)
            self.optim3d.step()
            critic_loss.backward(inputs=list(self.critic.parameters()))
            self.optim_critic.step()

            self.after_optim_step()  # overwrite to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim3d.zero_grad()
            self.optim_critic.zero_grad()

            self.optim_steps += 1
        return peasant_loss, philosopher_loss, critic_loss, view2d.detach(), view3d.detach()

    def predict(self, data_loader: DataLoader, epoch: int = 0, optim: torch.optim.Optimizer = None,
                return_predictions: bool = False) -> Tuple[Dict, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """
        get predictions for data in dataloader and do backpropagation if an optimizer is provided
        Args:
            data_loader: pytorch dataloader from which the batches will be taken
            epoch: optional parameter for logging
            optim: pytorch optimizer. If this is none, no backpropagation is done
            return_predictions: return the prdictions if true, else returns None

        Returns:
            metrics: a dictionary with all the metrics and the loss
            predictions: all predictions in the epoch
            targets: all targets of the epoch
        """
        args = self.args
        total_metrics = {k: 0 for k in self.evaluate_metrics(torch.ones((2, 2), device=self.device),
                                                             torch.ones((2, 2), device=self.device)).keys()}
        total_metrics[type(self.loss_func).__name__] = 0
        total_metrics[type(self.critic_loss_func).__name__] = 0
        total_metrics['philosopher_loss'] = 0
        epoch_targets = []
        epoch_predictions = []
        for i, batch in enumerate(data_loader):
            batch = [element.to(self.device) for element in batch]
            peasant_loss, philosopher_loss, critic_loss, predictions, targets = self.process_batch(batch, optim)

            with torch.no_grad():
                if self.optim_steps % args.log_iterations == args.log_iterations - 1 and optim != None:  # log every log_iterations during train
                    metrics_results = self.evaluate_metrics(predictions, targets, batch)
                    metrics_results[type(self.loss_func).__name__] = peasant_loss.item()
                    metrics_results[type(self.critic_loss_func).__name__] = critic_loss.item()
                    metrics_results['philosopher_loss'] = philosopher_loss.item()
                    self.run_tensorboard_functions(predictions, targets, step=self.optim_steps, data_split='train')
                    self.tensorboard_log(metrics_results, data_split='train', step=self.optim_steps, epoch=epoch)
                    print('[Epoch %d; Iter %5d/%5d] %s: peasant_loss: %.7f' % (epoch,
                                                                               i + 1, len(data_loader), 'train',
                                                                               peasant_loss.item()))
                if optim == None:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics_results = self.evaluate_metrics(predictions, targets, batch)
                    metrics_results[type(self.loss_func).__name__] = peasant_loss.item()
                    metrics_results[type(self.critic_loss_func).__name__] = critic_loss.item()
                    metrics_results['philosopher_loss'] = philosopher_loss.item()
                    self.run_tensorboard_functions(predictions, targets, step=self.optim_steps, data_split='val')
                    for key, value in metrics_results.items():
                        total_metrics[key] += value
                if return_predictions:
                    epoch_predictions.append(predictions.detach().cpu())
                    epoch_targets.append(targets.detach().cpu())

        total_metrics = {k: v / len(data_loader) for k, v in total_metrics.items()}
        epoch_predictions = torch.cat(epoch_predictions, dim=0) if return_predictions else None
        epoch_targets = torch.cat(epoch_targets, dim=0) if return_predictions else None
        return total_metrics, epoch_predictions, epoch_targets

    def initialize_optimizer(self, optim):
        normal_params = [v for k, v in self.model.named_parameters() if not 'batch_norm' in k]
        batch_norm_params = [v for k, v in self.model.named_parameters() if 'batch_norm' in k]
        self.optim = optim([{'params': batch_norm_params, 'weight_decay': 0},
                            {'params': normal_params}],
                           **self.args.optimizer_params)
        normal_params3d = [v for k, v in self.model3d.named_parameters() if not 'batch_norm' in k]
        batch_norm_params3d = [v for k, v in self.model3d.named_parameters() if 'batch_norm' in k]
        self.optim3d = optim([{'params': batch_norm_params3d, 'weight_decay': 0},
                              {'params': normal_params3d}],
                             **self.args.optimizer_params)
        normal_params_critic = [v for k, v in self.critic.named_parameters() if not 'batch_norm' in k]
        batch_norm_params_critic = [v for k, v in self.critic.named_parameters() if 'batch_norm' in k]
        self.optim_critic = optim([{'params': batch_norm_params_critic, 'weight_decay': 0},
                                   {'params': normal_params_critic}],
                                  **self.args.optimizer_params)

    def step_schedulers(self, metrics=None):
        try:
            self.lr_scheduler.step(metrics=metrics)
            self.lr_scheduler3d.step(metrics=metrics)
            self.lr_scheduler_critic.step(metrics=metrics)
        except:
            self.lr_scheduler.step()
            self.lr_scheduler3d.step()
            self.lr_scheduler_critic.step()

    def initialize_scheduler(self):
        if self.args.lr_scheduler:  # Needs "from torch.optim.lr_scheduler import *" to work
            self.lr_scheduler = globals()[self.args.lr_scheduler](self.optim, **self.args.lr_scheduler_params)
            self.lr_scheduler3d = globals()[self.args.lr_scheduler](self.optim3d, **self.args.lr_scheduler_params)
            self.lr_scheduler_critic = globals()[self.args.lr_scheduler](self.optim_critic,
                                                                         **self.args.lr_scheduler_params)
        else:
            self.lr_scheduler = None
