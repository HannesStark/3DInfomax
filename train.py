import argparse
import concurrent.futures
import copy
import os
import re

from icecream import install
from ogb.lsc import DglPCQM4MDataset, PCQM4MEvaluator
from ogb.utils import smiles2graph

from commons.utils import seed_all, get_random_indices, TENSORBOARD_FUNCTIONS
from datasets.ZINC_dataset import ZINCDataset
from datasets.bace_geomol_feat import BACEGeomol
from datasets.bace_geomol_featurization_of_qm9 import BACEGeomolQM9Featurization
from datasets.bace_geomol_random_split import BACEGeomolRandom
from datasets.bbbp_geomol_feat import BBBPGeomol
from datasets.bbbp_geomol_featurization_of_qm9 import BBBPGeomolQM9Featurization
from datasets.bbbp_geomol_random_split import BBBPGeomolRandom
from datasets.esol_geomol_feat import ESOLGeomol
from datasets.esol_geomol_featurization_of_qm9 import ESOLGeomolQM9Featurization
from datasets.file_loader_drugs import FileLoaderDrugs
from datasets.file_loader_qm9 import FileLoaderQM9
from datasets.geom_drugs_dataset import GEOMDrugs
from datasets.geom_qm9_dataset import GEOMqm9
from datasets.geomol_geom_qm9_dataset import QM9GeomolFeatDataset
from datasets.lipo_geomol_feat import LIPOGeomol
from datasets.lipo_geomol_featurization_of_qm9 import LIPOGeomolQM9Featurization
from datasets.ogbg_dataset_extension import OGBGDatasetExtension
from datasets.qm9_dataset_geomol_conformers import QM9DatasetGeomolConformers
from datasets.qm9_dataset_rdkit_conformers import QM9DatasetRDKITConformers

from datasets.qm9_geomol_featurization import QM9GeomolFeaturization
from datasets.qmugs_dataset import QMugsDataset
from models.geomol_mpnn import GeomolGNNWrapper
from trainer.byol_trainer import BYOLTrainer
from trainer.byol_wrapper import BYOLwrapper

import seaborn

from trainer.graphcl_trainer import GraphCLTrainer
from trainer.optimal_transport_trainer import OptimalTransportTrainer
from trainer.philosophy_trainer import PhilosophyTrainer
from trainer.self_supervised_ae_trainer import SelfSupervisedAETrainer

from trainer.self_supervised_alternating_trainer import SelfSupervisedAlternatingTrainer

from trainer.self_supervised_trainer import SelfSupervisedTrainer

import yaml
from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from datasets.samplers import *  # do not remove

from datasets.qm9_dataset import QM9Dataset
from torch.utils.data import DataLoader, Subset

from trainer.metrics import QM9DenormalizedL1, QM9DenormalizedL2, \
    QM9SingleTargetDenormalizedL1, Rsquared, NegativeSimilarity, MeanPredictorLoss, \
    PositiveSimilarity, ContrastiveAccuracy, TrueNegativeRate, TruePositiveRate, Alignment, Uniformity, \
    BatchVariance, DimensionCovariance, MAE, PositiveSimilarityMultiplePositivesSeparate2d, \
    NegativeSimilarityMultiplePositivesSeparate2d, OGBEvaluator, PearsonR, PositiveProb, NegativeProb, \
    Conformer2DVariance, Conformer3DVariance, PCQM4MEvaluatorWrapper
from trainer.trainer import Trainer

# turn on for debugging C code like Segmentation Faults
import faulthandler
faulthandler.enable()
install()
seaborn.set_theme()



def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/pna.yml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset', type=str, default='qm9', help='[qm9, zinc, drugs, geom_qm9, molhiv]')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--critic_loss', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--critic_loss_params', type=dict, default={},
                   help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--linear_probing_samples', type=int, default=500,
                   help='number of samples to use for linear probing in the run_eval_per_epoch function of the self supervised trainer')
    p.add_argument('--num_conformers', type=int, default=3,
                   help='number of conformers to use if we are using multiple conformers on the 3d side')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='mae_denormalized', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--pretrain_checkpoint', type=str, help='Specify path to finetune from a pretrained checkpoint')
    p.add_argument('--transfer_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--frozen_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--exclude_from_transfer', default=[],
                   help='parameters that usually should not be transferred like batchnorm params')
    p.add_argument('--transferred_lr', type=float, default=None, help='set to use a different LR for transfer layers')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--required_data', default=[],
                   help='what will be included in a batch like [dgl_graph, targets, dgl_graph3d]')
    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--use_e_features', default=True, type=bool, help='ignore edge features if set to False')
    p.add_argument('--targets', default=[], help='properties that should be predicted')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--dist_embedding', type=bool, default=False, help='add dist embedding to complete graphs edges')
    p.add_argument('--num_radial', type=int, default=6, help='number of frequencies for distance embedding')
    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--model3d_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--model3d_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--critic_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--critic_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--trainer', type=str, default='contrastive', help='[contrastive, byol, alternating, philosophy]')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--force_random_split', type=bool, default=False, help='use random split for ogb')
    p.add_argument('--reuse_pre_train_data', type=bool, default=False, help='use all data instead of ignoring that used during pre-training')
    p.add_argument('--transfer_3d', type=bool, default=False, help='set true to load the 3d network instead of the 2d network')
    return p.parse_args()


def get_trainer(args, model, data, device, metrics):
    tensorboard_functions = {function: TENSORBOARD_FUNCTIONS[function] for function in args.tensorboard_functions}
    if args.model3d_type:
        model3d = globals()[args.model3d_type](
            node_dim=0,  # 3d model has no input node features
            edge_dim=data[0][1].edata['d'].shape[
                1] if args.use_e_features and isinstance(data[0][1], dgl.DGLGraph) else 0,
            avg_d=data.avg_degree if hasattr(data, 'avg_degree') else 1,
            **args.model3d_parameters)
        print('3D model trainable params: ', sum(p.numel() for p in model3d.parameters() if p.requires_grad))

        critic = None
        if args.trainer == 'byol':
            ssl_trainer = BYOLTrainer
        elif args.trainer == 'alternating':
            ssl_trainer = SelfSupervisedAlternatingTrainer
        elif args.trainer == 'autoencoder':
            ssl_trainer = SelfSupervisedAETrainer
        elif args.trainer == 'contrastive':
            ssl_trainer = SelfSupervisedTrainer
        elif args.trainer == 'philosophy':
            ssl_trainer = PhilosophyTrainer
            critic = globals()[args.critic_type](**args.critic_parameters)
        return ssl_trainer(model=model, model3d=model3d, critic=critic, args=args, metrics=metrics,
                           main_metric=args.main_metric, main_metric_goal=args.main_metric_goal,
                           optim=globals()[args.optimizer], loss_func=globals()[args.loss_func](**args.loss_params),
                           critic_loss=globals()[args.critic_loss](**args.critic_loss_params), device=device,
                           tensorboard_functions=tensorboard_functions,
                           scheduler_step_per_batch=args.scheduler_step_per_batch)
    else:
        if args.trainer == 'optimal_transport':
            trainer = OptimalTransportTrainer
        elif args.trainer == 'graphcl_trainer':
            trainer = GraphCLTrainer
        else:
            trainer = Trainer
        return trainer(model=model, args=args, metrics=metrics, main_metric=args.main_metric,
                       main_metric_goal=args.main_metric_goal, optim=globals()[args.optimizer],
                       loss_func=globals()[args.loss_func](**args.loss_params), device=device,
                       tensorboard_functions=tensorboard_functions,
                       scheduler_step_per_batch=args.scheduler_step_per_batch)


def load_model(args, data, device):
    model = globals()[args.model_type](avg_d=data.avg_degree if hasattr(data, 'avg_degree') else 1, device=device,
                                       **args.model_parameters)
    if args.pretrain_checkpoint:
        # get arguments used during pretraining
        with open(os.path.join(os.path.dirname(args.pretrain_checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            pretrain_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        pretrain_args = argparse.Namespace()
        pretrain_args.__dict__.update(pretrain_dict)
        checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
        # get all the weights that have something from 'args.transfer_layers' in their keys name
        # but only if they do not contain 'teacher' and remove 'student.' which we need for loading from BYOLWrapper
        weights_key = 'model3d_state_dict' if args.transfer_3d == True else 'model_state_dict'
        pretrained_gnn_dict = {re.sub('^gnn\.|^gnn2\.', 'node_gnn.', k.replace('student.', '')): v
                               for k, v in checkpoint[weights_key].items() if any(
                transfer_layer in k for transfer_layer in args.transfer_layers) and 'teacher' not in k and not any(
                to_exclude in k for to_exclude in args.exclude_from_transfer)}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_gnn_dict)  # update the gnn layers with the pretrained weights
        model.load_state_dict(model_state_dict)
        if args.reuse_pre_train_data:
            return model, 0, pretrain_args.dataset == args.dataset
        else:
            return model, pretrain_args.num_train, pretrain_args.dataset == args.dataset
    return model, None, False


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    metrics_dict = {'rsquared': Rsquared(),
                    'mae': MAE(),
                    'pearsonr': PearsonR(),
                    'ogbg-molhiv': OGBEvaluator(d_name='ogbg-molhiv', metric='rocauc'),
                    'ogbg-molpcba': OGBEvaluator(d_name='ogbg-molpcba', metric='ap'),
                    'ogbg-molbace': OGBEvaluator(d_name='ogbg-molbace', metric='rocauc'),
                    'ogbg-molbbbp': OGBEvaluator(d_name='ogbg-molbbbp', metric='rocauc'),
                    'ogbg-molclintox': OGBEvaluator(d_name='ogbg-molclintox', metric='rocauc'),
                    'ogbg-moltoxcast': OGBEvaluator(d_name='ogbg-moltoxcast', metric='rocauc'),
                    'ogbg-moltox21': OGBEvaluator(d_name='ogbg-moltox21', metric='rocauc'),
                    'ogbg-mollipo': OGBEvaluator(d_name='ogbg-mollipo', metric='rmse'),
                    'ogbg-molmuv': OGBEvaluator(d_name='ogbg-molmuv', metric='ap'),
                    'ogbg-molsider': OGBEvaluator(d_name='ogbg-molsider', metric='rocauc'),
                    'ogbg-molfreesolv': OGBEvaluator(d_name='ogbg-molfreesolv', metric='rmse'),
                    'ogbg-molesol': OGBEvaluator(d_name='ogbg-molesol', metric='rmse'),
                    'pcqm4m': PCQM4MEvaluatorWrapper(),
                    'conformer_3d_variance': Conformer3DVariance(),
                    'conformer_2d_variance': Conformer2DVariance(),
                    'positive_similarity': PositiveSimilarity(),
                    'positive_similarity_multiple_positives_separate2d': PositiveSimilarityMultiplePositivesSeparate2d(),
                    'positive_prob': PositiveProb(),
                    'negative_prob': NegativeProb(),
                    'negative_similarity': NegativeSimilarity(),
                    'negative_similarity_multiple_positives_separate2d': NegativeSimilarityMultiplePositivesSeparate2d(),
                    'contrastive_accuracy': ContrastiveAccuracy(threshold=0.5009),
                    'true_negative_rate': TrueNegativeRate(threshold=0.5009),
                    'true_positive_rate': TruePositiveRate(threshold=0.5009),
                    'mean_predictor_loss': MeanPredictorLoss(globals()[args.loss_func](**args.loss_params)),
                    'uniformity': Uniformity(t=2),
                    'alignment': Alignment(alpha=2),
                    'batch_variance': BatchVariance(),
                    'dimension_covariance': DimensionCovariance()
                    }
    print('using device: ', device)
    if args.dataset == 'qm9' or args.dataset == 'qm9_rdkit'or args.dataset == 'qm9_neuralconf':
        return train_qm9(args, device, metrics_dict)
    elif args.dataset == 'zinc':
        return train_zinc(args, device, metrics_dict)
    elif args.dataset == 'qmugs':
        return train_geom(args, device, metrics_dict)
    elif args.dataset == 'drugs' or args.dataset == 'geom_qm9' or args.dataset == 'qm9_geomol_feat' or args.dataset == 'file_loader_drugs' or args.dataset == 'file_loader_qm9':
        return train_geom(args, device, metrics_dict)
    elif args.dataset == 'qm9_geomol':
        return train_qm9_geomol_featurization(args, device, metrics_dict)
    elif 'pcqm4m' == args.dataset:
        return train_pcqm4m(args, device, metrics_dict)
    elif 'geomol' in args.dataset:
        return train_geomol(args, device, metrics_dict)
    elif 'ogbg' in args.dataset:
        return train_ogbg(args, device, metrics_dict)


def train_geomol(args, device, metrics_dict):
    if args.dataset == 'bace_geomol':
        dataset = BACEGeomol
    elif args.dataset == 'bbbp_geomol':
        dataset = BBBPGeomol
    elif args.dataset == 'bace_geomol_random':
        dataset = BACEGeomolRandom
    elif args.dataset == 'bbbp_geomol_random':
        dataset = BBBPGeomolRandom
    elif args.dataset == 'esol_geomol':
        dataset = ESOLGeomol
    elif args.dataset == 'lipo_geomol':
        dataset = LIPOGeomol
    elif args.dataset == 'esol_geomol_qm9_featurization':
        dataset = ESOLGeomolQM9Featurization
    elif args.dataset == 'lipo_geomol_qm9_featurization':
        dataset = LIPOGeomolQM9Featurization
    if args.dataset == 'bace_geomol_qm9_featurization':
        dataset = BACEGeomolQM9Featurization
    elif args.dataset == 'bbbp_geomol_qm9_featurization':
        dataset = BBBPGeomolQM9Featurization

    train = dataset(split='train', device=device)
    val = dataset(split='val', device=device)
    test = dataset(split='test', device=device)

    model = globals()[args.model_type](node_dim=train[0][0].z.shape[1], edge_dim=train[0][0].edge_attr.shape[1],
                                       **args.model_parameters)

    if args.pretrain_checkpoint:
        checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
        pretrained_gnn_dict = {k.replace('student.', ''): v for k, v in checkpoint.items() if any(
            transfer_layer in k for transfer_layer in args.transfer_layers) and 'teacher' not in k and not any(
            to_exclude in k for to_exclude in args.exclude_from_transfer)}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_gnn_dict)  # update the gnn layers with the pretrained weights
        model.load_state_dict(model_state_dict)

    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_function)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate_function)
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_function)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    metric_name = [key for key in metrics_dict.keys() if 'ogbg-mol' + args.dataset.split('_')[0] == key.lower()][0]
    metrics[metric_name] = metrics_dict[metric_name]
    args.main_metric = metric_name
    args.val_per_batch = False
    args.main_metric_goal = 'min' if metrics[metric_name].metric == 'rmse' else 'max'
    trainer = get_trainer(args=args, model=model, data=train, device=device, metrics=metrics)
    val_metrics = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_metrics = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def train_qm9_geomol_featurization(args, device, metrics_dict):
    all_data = QM9GeomolFeaturization(return_types=args.required_data, target_tasks=args.targets, device=device,
                                      dist_embedding=args.dist_embedding, num_radial=args.num_radial)

    all_idx = get_random_indices(len(all_data), args.seed_data)
    model_idx = all_idx[:100000]
    test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(all_data))]
    val_idx = all_idx[len(model_idx) + len(test_idx):]
    train_idx = model_idx[:args.num_train]
    # for debugging purposes:
    # test_idx = all_idx[len(model_idx): len(model_idx) + 200]
    # val_idx = all_idx[len(model_idx) + len(test_idx): len(model_idx) + len(test_idx) + 3000]

    model = globals()[args.model_type](node_dim=all_data[0][0].z.shape[1], edge_dim=all_data[0][0].edge_attr.shape[1],
                                       **args.model_parameters)

    if args.pretrain_checkpoint:
        checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
        pretrained_gnn_dict = {k.replace('student.', ''): v for k, v in checkpoint.items() if any(
            transfer_layer in k for transfer_layer in args.transfer_layers) and 'teacher' not in k and not any(
            to_exclude in k for to_exclude in args.exclude_from_transfer)}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_gnn_dict)  # update the gnn layers with the pretrained weights
        model.load_state_dict(model_state_dict)

    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Training on {len(train_idx)} samples from the model sequences')
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)

    train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_function)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size, collate_fn=collate_function)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size, collate_fn=collate_function)

    metrics_dict.update({'mae_denormalized': QM9DenormalizedL1(dataset=all_data),
                         'mse_denormalized': QM9DenormalizedL2(dataset=all_data)})
    metrics = {metric: metrics_dict[metric] for metric in args.metrics if metric != 'qm9_properties'}
    if 'qm9_properties' in args.metrics:
        metrics.update(
            {task: QM9SingleTargetDenormalizedL1(dataset=all_data, task=task) for task in all_data.target_tasks})

    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    val_metrics = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_metrics = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def train_pcqm4m(args, device, metrics_dict):
    dataset = DglPCQM4MDataset(smiles2graph=smiles2graph)
    split_idx = dataset.get_idx_split()
    split_idx["train"] = split_idx["train"][:args.num_train]
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)

    train_loader = DataLoader(Subset(dataset, split_idx["train"]), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_function)
    val_loader = DataLoader(Subset(dataset, split_idx["valid"]), batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_function)
    test_loader = DataLoader(Subset(dataset, split_idx["test"]), batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_function)

    model, num_pretrain, transfer_from_same_dataset = load_model(args, data=dataset, device=device)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    metrics[args.dataset] = metrics_dict[args.dataset]
    args.main_metric = args.dataset
    args.main_metric_goal = 'min'
    trainer = get_trainer(args=args, model=model, data=dataset, device=device, metrics=metrics)
    val_metrics = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_metrics = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def train_ogbg(args, device, metrics_dict):
    dataset = OGBGDatasetExtension(return_types=args.required_data, device=device, name=args.dataset)
    split_idx = dataset.get_idx_split()
    if args.force_random_split == True:
        all_idx = get_random_indices(len(dataset), args.seed_data)
        split_idx["train"] = all_idx[:len(split_idx["train"])]
        split_idx["train"] = all_idx[len(split_idx["train"]):len(split_idx["train"])+len(split_idx["valid"])]
        split_idx["train"] = all_idx[len(split_idx["train"])+len(split_idx["valid"]):]
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)

    train_loader = DataLoader(Subset(dataset, split_idx["train"]), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_function)
    val_loader = DataLoader(Subset(dataset, split_idx["valid"]), batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_function)
    test_loader = DataLoader(Subset(dataset, split_idx["test"]), batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_function)

    model, num_pretrain, transfer_from_same_dataset = load_model(args, data=dataset, device=device)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    metrics[args.dataset] = metrics_dict[args.dataset]
    args.main_metric = args.dataset
    args.val_per_batch = False
    args.main_metric_goal = 'min' if metrics[args.main_metric].metric == 'rmse' else 'max'
    trainer = get_trainer(args=args, model=model, data=dataset, device=device, metrics=metrics)
    val_metrics = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_metrics = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def train_zinc(args, device, metrics_dict):
    train_data = ZINCDataset(split='train', device=device)
    val_data = ZINCDataset(split='val', device=device)
    test_data = ZINCDataset(split='test', device=device)

    model, num_pretrain, transfer_from_same_dataset = load_model(args, data=train_data, device=device)
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)
    if args.train_sampler != None:
        sampler = globals()[args.train_sampler](data_source=train_data, batch_size=args.batch_size,
                                                indices=range(len(train_data)))
        train_loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_function)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_function)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_function)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    trainer = get_trainer(args=args, model=model, data=train_data, device=device, metrics=metrics)
    val_metrics = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_metrics = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def train_geom(args, device, metrics_dict):
    if args.dataset == 'drugs':
        dataset = GEOMDrugs
    elif args.dataset == 'geom_qm9':
        dataset = GEOMqm9
    elif args.dataset == 'qmugs':
        dataset = QMugsDataset
    elif args.dataset == 'qm9_geomol_feat':
        dataset = QM9GeomolFeatDataset
    elif args.dataset == 'file_loader_drugs':
        dataset = FileLoaderDrugs
    elif args.dataset == 'file_loader_qm9':
        dataset = FileLoaderQM9
    all_data = dataset(return_types=args.required_data, target_tasks=args.targets, device=device,
                       num_conformers=args.num_conformers)
    all_idx = get_random_indices(len(all_data), args.seed_data)
    if args.dataset == 'drugs':
        model_idx = all_idx[:280000]  # 304293 in all data
    elif args.dataset in ['geom_qm9', 'qm9_geomol_feat']:
        model_idx = all_idx[:100000]
    elif args.dataset == 'qmugs':
        model_idx = all_idx[:620000]
    elif args.dataset == 'file_loader_qm9':
        model_idx = all_idx[:80000]  # 107857 molecules in all_data
    elif args.dataset == 'file_loader_drugs':
        model_idx = all_idx[:160000]
    test_idx = all_idx[len(model_idx): len(model_idx) + int(0.05 * len(all_data))]
    if args.dataset in ['file_loader_drugs', 'file_loader_qm9']:
        val_idx = all_idx[max(len(model_idx) + len(test_idx), len(all_data) - 1000):]
    else:
        val_idx = all_idx[len(model_idx) + len(test_idx):]
    train_idx = model_idx[:args.num_train]
    # for debugging purposes:
    # test_idx = all_idx[len(model_idx): len(model_idx) + 200]
    # val_idx = all_idx[len(model_idx) + len(test_idx): len(model_idx) + len(test_idx) + 3000]

    model, num_pretrain, transfer_from_same_dataset = load_model(args, data=all_data, device=device)
    if transfer_from_same_dataset:
        train_idx = model_idx[num_pretrain: num_pretrain + args.num_train]
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Training on {len(train_idx)} samples from the model sequences')
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)

    if args.train_sampler != None:
        sampler = globals()[args.train_sampler](data_source=all_data, batch_size=args.batch_size, indices=train_idx)
        train_loader = DataLoader(Subset(all_data, train_idx), batch_sampler=sampler, collate_fn=collate_function)
    else:
        train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_function)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size, collate_fn=collate_function)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size, collate_fn=collate_function)

    if 'mae_denormalized' in args.metrics:
        metrics_dict.update({'mae_denormalized': QM9DenormalizedL1(dataset=all_data),
                             'mse_denormalized': QM9DenormalizedL2(dataset=all_data)})
    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    val_metrics = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_metrics = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def train_qm9(args, device, metrics_dict):
    if args.dataset == 'qm9_rdkit':
        all_data = QM9DatasetRDKITConformers(return_types=args.required_data, target_tasks=args.targets, device=device,
                              dist_embedding=args.dist_embedding, num_radial=args.num_radial)
    elif args.dataset == 'qm9_neuralconf':

        all_data = QM9DatasetGeomolConformers(return_types=args.required_data, target_tasks=args.targets, device=device,
                              dist_embedding=args.dist_embedding, num_radial=args.num_radial)
    else:
        all_data = QM9Dataset(return_types=args.required_data, target_tasks=args.targets, device=device,
                          dist_embedding=args.dist_embedding, num_radial=args.num_radial)

    all_idx = get_random_indices(len(all_data), args.seed_data)
    model_idx = all_idx[:100000]
    test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(all_data))]
    val_idx = all_idx[len(model_idx) + len(test_idx):]
    train_idx = model_idx[:args.num_train]

    if args.num_val != None:
        train_idx = all_idx[:args.num_train]
        val_idx = all_idx[len(train_idx): len(train_idx) + args.num_val]
        test_idx = all_idx[len(train_idx) + args.num_val: len(train_idx) + 2*args.num_val]
    # for debugging purposes:
    # test_idx = all_idx[len(model_idx): len(model_idx) + 20]
    # val_idx = all_idx[len(model_idx) + len(test_idx): len(model_idx) + len(test_idx) + 30]

    model, num_pretrain, transfer_from_same_dataset = load_model(args, data=all_data, device=device)
    if transfer_from_same_dataset:
        train_idx = model_idx[num_pretrain: num_pretrain + args.num_train]
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


    print(f'Training on {len(train_idx)} samples from the model sequences')
    print(f'Validating on {len(val_idx)} samples')
    print(f'Testing on {len(test_idx)} samples')
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)
    if args.train_sampler != None:
        sampler = globals()[args.train_sampler](data_source=all_data, batch_size=args.batch_size, indices=train_idx)
        train_loader = DataLoader(Subset(all_data, train_idx), batch_sampler=sampler, collate_fn=collate_function)
    else:
        train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_function)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size, collate_fn=collate_function)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size, collate_fn=collate_function)

    metrics_dict.update({'mae_denormalized': QM9DenormalizedL1(dataset=all_data),
                         'mse_denormalized': QM9DenormalizedL2(dataset=all_data)})
    metrics = {metric: metrics_dict[metric] for metric in args.metrics if metric != 'qm9_properties'}
    if 'qm9_properties' in args.metrics:
        metrics.update(
            {task: QM9SingleTargetDenormalizedL1(dataset=all_data, task=task) for task in all_data.target_tasks})

    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    val_metrics = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_metrics = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def get_arguments():
    args = parse_arguments()
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
    args = get_arguments()

    if args.multithreaded_seeds != []:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for seed in args.multithreaded_seeds:
                args_copy = get_arguments()
                args_copy.seed = seed
                futures.append(executor.submit(train, args_copy))
            results = [f.result() for f in
                       futures]  # list of tuples of dictionaries with the validation results first and the test results second
        all_val_metrics = defaultdict(list)
        all_test_metrics = defaultdict(list)
        log_dirs = []
        for result in results:
            val_metrics, test_metrics, log_dir = result
            log_dirs.append(log_dir)
            for key in val_metrics.keys():
                all_val_metrics[key].append(val_metrics[key])
                all_test_metrics[key].append(test_metrics[key])
        files = [open(os.path.join(dir, 'multiple_seed_validation_statistics.txt'), 'w') for dir in log_dirs]
        print('Validation results:')
        for key, value in all_val_metrics.items():
            metric = np.array(value)
            for file in files:
                file.write(f'\n{key:}\n')
                file.write(f'mean: {metric.mean()}\n')
                file.write(f'stddev: {metric.std()}\n')
                file.write(f'stderr: {metric.std() / np.sqrt(len(metric))}\n')
                file.write(f'values: {value}\n')
            print(f'\n{key}:')
            print(f'mean: {metric.mean()}')
            print(f'stddev: {metric.std()}')
            print(f'stderr: {metric.std() / np.sqrt(len(metric))}')
            print(f'values: {value}')
        for file in files:
            file.close()
        files = [open(os.path.join(dir, 'multiple_seed_test_statistics.txt'), 'w') for dir in log_dirs]
        print('Test results:')
        for key, value in all_test_metrics.items():
            metric = np.array(value)
            for file in files:
                file.write(f'\n{key:}\n')
                file.write(f'mean: {metric.mean()}\n')
                file.write(f'stddev: {metric.std()}\n')
                file.write(f'stderr: {metric.std() / np.sqrt(len(metric))}\n')
                file.write(f'values: {value}\n')
            print(f'\n{key}:')
            print(f'mean: {metric.mean()}')
            print(f'stddev: {metric.std()}')
            print(f'stderr: {metric.std() / np.sqrt(len(metric))}')
            print(f'values: {value}')
        for file in files:
            file.close()
    else:
        train(args)
