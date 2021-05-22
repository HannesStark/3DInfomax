import argparse
import os
from icecream import install

from commons.utils import seed_all, get_random_indices
from trainer.byol_trainer import BYOLTrainer
from trainer.byol_wrapper import BYOLwrapper

install()

from trainer.self_supervised_trainer import SelfSupervisedTrainer

import torch
import yaml
from datasets.custom_collate import * # do not remove
from models import * # do not remove
from torch.nn import * # do not remove
from torch.optim import *
from commons.contrastive_loss import * # do not remove
from torch.optim.lr_scheduler import *
from datasets.samplers import * # do not remove

from datasets.qm9_dataset import QM9Dataset
from torch.utils.data import DataLoader, Subset


from trainer.metrics import QM9DenormalizedL1, QM9DenormalizedL2, pearsonr, \
    QM9SingleTargetDenormalizedL1, Rsquared, NegativeSimilarity, MeanPredictorLoss, \
    F1Contrastive, PositiveSimilarity, ContrastiveAccuracy, TrueNegativeRate, TruePositiveRate
from trainer.trainer import Trainer


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print('using device: ', device)
    # will only return the target task properties so only those will be predicted
    all_data = QM9Dataset(return_types=args.required_data, features=args.features, features3d=args.features3d,
                          e_features=args.e_features,
                          e_features3d=args.e_features3d, pos_dir=args.pos_dir,
                          target_tasks=args.targets,
                          dist_embedding=args.dist_embedding, num_radial=args.num_radial,
                          prefetch_graphs=args.prefetch_graphs)

    all_idx = get_random_indices(len(all_data), args.seed_data)
    model_idx = all_idx[:100000]
    test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(all_data))]
    val_idx = all_idx[len(model_idx) + len(test_idx):]
    train_idx = model_idx[:args.num_train]
    # TODO REMOVE debugging stuff:
    # test_idx = all_idx[len(model_idx): len(model_idx) + 200]
    # val_idx = all_idx[len(model_idx) + len(test_idx): len(model_idx) + len(test_idx)]

    model = globals()[args.model_type](node_dim=all_data[0][0].ndata['f'].shape[1],
                                       edge_dim=all_data[0][0].edata['w'].shape[
                                           1] if args.use_e_features else 0,
                                       avg_d=all_data.avg_degree,
                                       **args.model_parameters)

    if args.pretrain_checkpoint:
        # get arguments used during pretraining
        with open(os.path.join(os.path.dirname(args.pretrain_checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            pretrain_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        pretrain_args = argparse.Namespace()
        pretrain_args.__dict__.update(pretrain_dict)
        train_idx = model_idx[pretrain_args.num_train: pretrain_args.num_train + args.num_train]

        checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
        # get all the weights that have something from 'args.transfer_layers' in their keys name
        # but only if they do not contain 'teacher' and remove 'student.' which we need for loading from BYOLWrapper
        pretrained_gnn_dict = {k.replace('student.', ''): v for k, v in checkpoint['model_state_dict'].items() if any(
            transfer_layer in k for transfer_layer in args.transfer_layers) and 'teacher' not in k}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_gnn_dict)  # update the gnn layers with the pretrained weights
        model.load_state_dict(model_state_dict)
    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Training on {len(train_idx)} samples from the model sequences')
    collate_function = globals()[args.collate_function]

    if args.train_sampler != None:
        sampler = globals()[args.train_sampler](data_source=all_data, batch_size=args.batch_size, indices=train_idx)
        train_loader = DataLoader(Subset(all_data, train_idx), batch_sampler=sampler, collate_fn=collate_function)
    else:
        train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_function)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size, collate_fn=collate_function)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size, collate_fn=collate_function)

    metrics_dict = {'mae_denormalized': QM9DenormalizedL1(dataset=all_data),
                    'mse_denormalized': QM9DenormalizedL2(dataset=all_data),
                    'pearsonr': pearsonr,
                    'rsquared': Rsquared(),
                    'positive_similarity': PositiveSimilarity(),
                    'negative_similarity': NegativeSimilarity(),
                    'f1_contrastive': F1Contrastive(threshold=0.5, device=device),
                    'contrastive_accuracy': ContrastiveAccuracy(threshold=0.5),
                    'true_negative_rate': TrueNegativeRate(threshold=0.5),
                    'true_positive_rate': TruePositiveRate(threshold=0.5),
                    'mean_predictor_loss': MeanPredictorLoss(globals()[args.loss_func](**args.loss_params)),
                    }
    metrics = {metric: metrics_dict[metric] for metric in args.metrics if metric != 'qm9_properties'}
    if 'qm9_properties' in args.metrics:
        metrics.update(
            {task: QM9SingleTargetDenormalizedL1(dataset=all_data, task=task) for task in all_data.target_tasks})

    # Needs "from torch.optim import *" and "from models import *" to work
    if args.model3d_type:
        model3d = globals()[args.model3d_type](node_dim=all_data[0][1].ndata['f'].shape[1],
                                               edge_dim=all_data[0][1].edata['w'].shape[
                                                   1] if args.use_e_features else 0,
                                               avg_d=all_data.avg_degree,
                                               **args.model3d_parameters)
        optim = globals()[args.optimizer](list(model.parameters()) + list(model3d.parameters()),
                                          **args.optimizer_params)
        ssl_trainer = BYOLTrainer if args.ssl_mode == 'byol' else SelfSupervisedTrainer
        trainer = ssl_trainer(model=model,
                              model3d=model3d,
                              args=args,
                              metrics=metrics,
                              main_metric=args.main_metric,
                              main_metric_goal=args.main_metric_goal,
                              optim=optim,
                              loss_func=globals()[args.loss_func](**args.loss_params),
                              device=device,
                              scheduler_step_per_batch=args.scheduler_step_per_batch)
    else:
        transferred_params = [v for k, v in model.named_parameters() if
                              any(transfer_name in k for transfer_name in args.transfer_layers)]
        new_params = [v for k, v in model.named_parameters() if
                      all(transfer_name not in k for transfer_name in args.transfer_layers)]
        transfer_lr = args.optimizer_params['lr'] if args.transferred_lr == None else args.transferred_lr
        optim = globals()[args.optimizer]([{'params': new_params},
                                           {'params': transferred_params, 'lr': transfer_lr}], **args.optimizer_params)
        trainer = Trainer(model=model,
                          args=args,
                          metrics=metrics,
                          main_metric=args.main_metric,
                          main_metric_goal=args.main_metric_goal,
                          optim=optim,
                          loss_func=globals()[args.loss_func](**args.loss_params),
                          device=device,
                          scheduler_step_per_batch=args.scheduler_step_per_batch)
    trainer.train(train_loader, val_loader)

    if args.eval_on_test:
        trainer.evaluation(test_loader, data_split='test')


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/9.yml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--prefetch_graphs', type=bool, default=True,
                   help='load graphs into memory (needs RAM and upfront computation) for faster data loading during training')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--num_train', type=int, default=100000, help='n samples of the model samples to use for train')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='mae_denormalized', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--pretrain_checkpoint', type=str, help='Specify path to finetune from a pretrained checkpoint')
    p.add_argument('--transfer_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--transferred_lr', type=float, default=None, help='set to use a different LR for transfer layers')

    p.add_argument('--features', default=[], help='types of input features like [atom_one_hot, hybridizations]')
    p.add_argument('--features3d', default=[],
                   help='types of input features like [atom_one_hot, hybridizations, constant_ones] but returned when appending 3d to the names in required data')
    p.add_argument('--e_features', default=[], help='types of input features like [atom_one_hot, hybridizations]')
    p.add_argument('--e_features3d', default=[],
                   help='types of input features like [atom_one_hot, hybridizations, constant_ones] but returned when appending 3d to the names in required data')

    p.add_argument('--pos_dir', type=bool, default=False, help='adds pos dir as key to dgl graphs (required for dgn)')
    p.add_argument('--required_data', default=[],
                   help='what will be included in a batch like [mol_graph, targets, mol_graph3d]')
    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--use_e_features', default=True, type=bool, help='ignore edge features if set to False')
    p.add_argument('--targets', default=[], help='properties that should be predicted')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--dist_embedding', type=bool, default=False, help='add dist embedding to complete graphs edges')
    p.add_argument('--num_radial', type=int, default=6, help='number of frequencies for distance embedding')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--model3d_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--model3d_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--ssl_mode', type=str, default='contrastive', help='[contrastive, byol]')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')

    args = p.parse_args()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}

    if args.checkpoint:  # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
    return args


if __name__ == '__main__':
    train(parse_arguments())
