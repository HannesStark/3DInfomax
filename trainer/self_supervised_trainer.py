import os
from itertools import chain
from typing import Dict, Callable

import dgl
import torch

from trainer.trainer import Trainer


class SelfSupervisedTrainer(Trainer):
    def __init__(self, model, model3d, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        self.model3d = model3d.to(device)  # move to device before loading optim params in super class
        super(SelfSupervisedTrainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions,
                                                    optim, main_metric_goal, loss_func, scheduler_step_per_batch)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model3d.load_state_dict(checkpoint['model3d_state_dict'])

    def forward_pass(self, batch):
        info2d, info3d, *snorm_n = tuple(batch)
        view2d = self.model(*info2d, *snorm_n)  # foward the rest of the batch to the model
        view3d = self.model3d(*info3d)
        loss = self.loss_func(view2d, view3d, nodes_per_graph=info2d[0].batch_num_nodes() if isinstance(info2d[0], dgl.DGLGraph) else None)
        return loss, view2d, view3d

    def evaluate_metrics(self, z2d, z3d, batch=None, val=False) -> Dict[str, float]:
        metric_results = {}
        metric_results[f'mean_pred'] = torch.mean(z2d).item()
        metric_results[f'std_pred'] = torch.std(z2d).item()
        metric_results[f'mean_targets'] = torch.mean(z3d).item()
        metric_results[f'std_targets'] = torch.std(z3d).item()
        if 'Local' in type(self.loss_func).__name__ and batch != None:
            node_indices = torch.cumsum(batch[0].batch_num_nodes(), dim=0)
            pos_mask = torch.zeros((len(z2d), len(z3d)), device=z2d.device)
            for graph_idx in range(1, len(node_indices)):
                pos_mask[node_indices[graph_idx - 1]: node_indices[graph_idx], graph_idx] = 1.
            pos_mask[0:node_indices[0], 0] = 1
            for key, metric in self.metrics.items():
                if not hasattr(metric, 'val_only') or val:
                    metric_results[key] = metric(z2d, z3d, pos_mask).item()
        else:
            for key, metric in self.metrics.items():
                if not hasattr(metric, 'val_only') or val:
                    metric_results[key] = metric(z2d, z3d).item()
        return metric_results

    def run_per_epoch_evaluations(self, data_loader):
        print('fitting linear probe')
        representations = []
        targets = []
        for batch in data_loader:
            batch = [element.to(self.device) for element in batch]
            loss, view2d, view3d = self.process_batch(batch, optim=None)
            representations.append(view2d)
            targets.append(batch[-1])
            if len(representations) * len(view2d) >= self.args.linear_probing_samples:
                break
        representations = torch.cat(representations, dim=0)
        targets = torch.cat(targets, dim=0)
        if len(representations) >= representations.shape[-1]:
            X, _ = torch.lstsq(targets, representations)
            X, _ = torch.lstsq(targets, representations)
            sol = X[:representations.shape[-1]]
            pred = representations @ sol
            mean_absolute_error = (pred - targets).abs().mean()
            self.writer.add_scalar('linear_probe_mae', mean_absolute_error.item(), self.optim_steps)
        else:
            raise ValueError(
                f'We have less linear_probing_samples {len(representations)} than the metric dimension {representations.shape[-1]}. Linear probing cannot be used.')

        print('finish fitting linear probe')

    def initialize_optimizer(self, optim):
        normal_params = [v for k, v in chain(self.model.named_parameters(), self.model3d.named_parameters()) if
                         not 'batch_norm' in k]
        batch_norm_params = [v for k, v in chain(self.model.named_parameters(), self.model3d.named_parameters()) if
                             'batch_norm' in k]

        self.optim = optim([{'params': batch_norm_params, 'weight_decay': 0},
                            {'params': normal_params}],
                           **self.args.optimizer_params)

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'model3d_state_dict': self.model3d.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))
