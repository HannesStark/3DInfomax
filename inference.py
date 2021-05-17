import copy

import torch

import selfsupervised3d
from selfsupervised3d.models import *
import os
import argparse
import yaml
from torch.utils.data import DataLoader

from commons import seed_all
from datasets.qm9_dataset import QM9Dataset
from trainer.metrics import QM9DenormalizedL1, pearsonr, QM9DenormalizedL2, \
    QM9SingleTargetDenormalizedL1
from trainer.trainer import Trainer


def inference(args):
    seed_all(args.seed)
    # will only return the target task properties so only those will be predicted
    all_data = QM9Dataset(return_types=['mol_graph', 'targets'], target_tasks=args.targets)

    # get data splits
    n_train = 100000
    n_test = int(0.1 * len(all_data))
    n_val = len(all_data) - n_train - n_test
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(all_data, [n_train, n_val, n_test])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=all_data.graph_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=all_data.graph_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=all_data.graph_collate)

    # Needs "from models import *" to work
    model = globals()[args.model_type](node_dim=all_data[0][0].ndata['f'].shape[1],
                                       edge_dim=all_data[0][0].edata['w'].shape[1],
                                       target_dim=all_data[0][1].shape[0],
                                       avg_d=all_data.avg_degree,
                                       **args.model_parameters)
    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    metrics = {
        'mae_denormalized': QM9DenormalizedL1(all_data.denormalize),
        'mse_denormalized': QM9DenormalizedL2(all_data.denormalize),
        'pearsonr': pearsonr}
    metrics.update(
        {task: QM9SingleTargetDenormalizedL1(dataset=all_data, task=task) for task in all_data.target_tasks})

    # Needs "from torch.optim import *" and "from models import *" to work
    trainer = Trainer(model, args,
                      metrics=metrics,
                      main_metric='mae_denormalized',
                      main_metric_goal=args.main_metric_goal,
                      optim=globals()[args.optimizer],
                      loss_func=globals()[args.loss_func],
                      scheduler_step_per_batch=args.scheduler_step_per_batch)

    trainer.evaluation(test_loader, data_split='val')


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/inference.yml')
    p.add_argument('--checkpoints_list', default=[], help='if there are paths specified here, they all are evaluated')
    p.add_argument('--batch_size', type=int, default=96, help='samples that will be processed in parallel')
    p.add_argument('--log_iterations', type=int, default=10, help='log every log_iterations (-1 for no logging)')

    args = p.parse_args()
    arg_dict = args.__dict__
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


if __name__ == '__main__':
    project_path = Path(os.path.dirname(selfsupervised3d.__path__[0]))
    os.chdir(project_path)
    original_args = copy.copy(parse_arguments())

    for checkpoint in original_args.checkpoints_list:
        args = copy.copy(original_args)
        arg_dict = args.__dict__
        arg_dict['checkpoint'] = checkpoint
        # overwrite the args from the checkpoint with the args from the inference config if they intersect
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as checkpoint_path:
            checkpoint_dict = yaml.load(checkpoint_path, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in args.__dict__.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
        # call teh actual inference
        inference(args)
