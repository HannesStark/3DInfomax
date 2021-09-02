import os
from argparse import Namespace

import numpy as np
import torch
import yaml
from icecream import ic
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from datasets.samplers import *  # do not remove
import seaborn as sn

sn.set_theme()
from commons.utils import get_random_indices
from datasets.geom_drugs_dataset import GEOMDrugs
from datasets.geom_qm9_dataset import GEOMqm9
from datasets.qm9_dataset import QM9Dataset
from datasets.qmugs_dataset import QMugsDataset

checkpoints = [
    'runs/PNA_qm9_NTXent_batchsize500_123_29-08_09-40-39/best_checkpoint.pt',
]

device = 'cuda'
device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
def singular_value_plot(checkpoint, i):
    args = Namespace()
    arg_dict = args.__dict__
    with open(os.path.join(os.path.dirname(checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
        checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        arg_dict.update(checkpoint_dict)


    if args.dataset == 'qm9':
        all_data = QM9Dataset(return_types=args.required_data, target_tasks=args.targets, device=device,
                              dist_embedding=args.dist_embedding, num_radial=args.num_radial)

        all_idx = get_random_indices(len(all_data), args.seed_data)
        model_idx = all_idx[:100000]
        test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(all_data))]
        val_idx = all_idx[len(model_idx) + len(test_idx):]
        train_idx = model_idx[:args.num_train]

    else:
        if args.dataset == 'drugs':
            dataset = GEOMDrugs
        elif args.dataset == 'geom_qm9':
            dataset = GEOMqm9
        elif args.dataset == 'qmugs':
            dataset = QMugsDataset
        all_data = dataset(return_types=args.required_data, target_tasks=args.targets, device=device,
                           num_conformers=args.num_conformers)
        all_idx = get_random_indices(len(all_data), args.seed_data)
        if args.dataset == 'drugs':
            model_idx = all_idx[:280000]  # 304293 in all data
        elif args.dataset in ['geom_qm9', 'qm9_geomol_feat']:
            model_idx = all_idx[:100000]
        elif args.dataset == 'qmugs':
            model_idx = all_idx[:620000]
        test_idx = all_idx[len(model_idx): len(model_idx) + int(0.05 * len(all_data))]
        val_idx = all_idx[len(model_idx) + len(test_idx):]
        train_idx = model_idx[:args.num_train]

    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)
    if args.train_sampler != None:
        sampler = globals()[args.train_sampler](data_source=all_data, batch_size=args.batch_size, indices=train_idx)
        train_loader = DataLoader(Subset(all_data, train_idx), batch_sampler=sampler, collate_fn=collate_function)
    else:
        train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_function)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_function)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_function)

    model = globals()[args.model_type](avg_d=all_data.avg_degree if hasattr(all_data, 'avg_degree') else 1,
                                       device=device,
                                       **args.model_parameters)
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])

    train_batch = next(iter(train_loader))
    with torch.no_grad():
        info2d, info3d, *snorm_n = tuple(train_batch)
        predictions = model(*info2d)

        u, s, v = torch.pca_lowrank(predictions.detach().cpu(), q=min(predictions.shape))

        s = 100 * s / s.sum()
        # plt.plot(s.numpy())
        plt.plot(np.cumsum(s.numpy()), label=os.path.split(checkpoint)[0])


for i, checkpoint in tqdm(enumerate(checkpoints)):
    singular_value_plot(checkpoint, i)

A = torch.randn((500, 256))
u, s, v = torch.pca_lowrank(A, q=min(A.shape))
s = 100 * s / s.sum()
plt.legend()
plt.show()
