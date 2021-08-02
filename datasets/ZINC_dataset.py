import os


import torch
import dgl
from torch.utils.data import Dataset
from tqdm import tqdm


class ZINCDataset(Dataset):
    """ The ZINC dataset as found here https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/molecules/prepare_molecules.ipynb

    """

    def __init__(self, split, device='cuda:0', normalize=False, prefetch_graphs=True, **kwargs):

        self.zinc_directory = 'dataset/ZINC'
        self.normalize = normalize
        self.device = device

        data_dict = torch.load(os.path.join(self.zinc_directory, split + '.pt'))

        self.meta_dict = {k: data_dict[k] for k in ('edge_slices', 'atom_slices')}
        self.edge_indices = data_dict['edge_indices']

        self.prefetch_graphs = prefetch_graphs
        if self.prefetch_graphs:
            print(
                'Load molecular graphs into memory (set prefetch_graphs to False to load them on the fly => slower training)')
            self.dgl_graphs = []
            for idx in tqdm(range(len(self.meta_dict['edge_slices']) - 1)):
                e_start = self.meta_dict['edge_slices'][idx]
                e_end = self.meta_dict['edge_slices'][idx + 1]
                edge_indices = self.edge_indices[:, e_start: e_end]
                self.dgl_graphs.append(dgl.graph((edge_indices[0], edge_indices[1])))
        print('Finish loading data into memory')

        self.atomic_number_onehot = data_dict['atomic_number_onehot']
        self.bond_type_onehot = data_dict['bond_type_onehot']
        self.targets = data_dict['targets'][:, None]
        if self.normalize:
            self.targets_mean = self.targets.mean(dim=0)
            self.targets_std = self.targets.std(dim=0)
            self.targets = ((self.targets - self.targets_mean) / self.targets_std)
            self.targets_mean = self.targets_mean.to(device)
            self.targets_std = self.targets_std.to(device)

    def __len__(self):
        return len(self.meta_dict['atom_slices']) - 1

    def __getitem__(self, idx):
        """

        Parameters
        ----------
        idx: integer between 0 and len(self) - 1

        Returns
        -------
        tuple of all data specified via the return_types parameter of the constructor
        """
        data = []
        e_start = self.meta_dict['edge_slices'][idx]
        e_end = self.meta_dict['edge_slices'][idx + 1]
        start = self.meta_dict['atom_slices'][idx]
        end = self.meta_dict['atom_slices'][idx + 1]

        if self.prefetch_graphs:
            g: dgl.DGLGraph = self.dgl_graphs[idx].to(self.device)
        else:
            edge_indices = self.edge_indices[:, e_start: e_end]
            g = dgl.graph((edge_indices[0], edge_indices[1])).to(self.device)
        g.ndata['f'] = self.atomic_number_onehot[start: end].to(self.device)
        g.edata['w'] = self.bond_type_onehot[e_start: e_end].to(self.device)
        return g, self.targets[idx]
