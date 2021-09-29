import copy
import os

import torch
import dgl
import torch_geometric
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector, get_atom_feature_dims, \
    get_bond_feature_dims
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from scipy.constants import physical_constants

from commons.spherical_encoding import dist_emb

hartree2eV = physical_constants['hartree-electron volt relationship'][0]


class QMugsDataset(Dataset):

    def __init__(self, return_types: list = None, target_tasks: list = None, normalize: bool = True, device='cuda:0',
                 num_conformers: int = 1, **kwargs):

        self.root = '../QMugs'
        self.processed_file = 'processed.pt'
        self.raw_csv = 'summary.csv'
        self.normalize = normalize
        self.device = device
        self.num_conformers = num_conformers
        self.return_types: list = return_types

        # load the data and get normalization values
        if not os.path.exists(os.path.join(self.root, 'processed', self.processed_file)):
            self.process()
        print('loading')
        data_dict = torch.load(os.path.join(self.root, 'processed', self.processed_file))
        print('finish loading')

        self.features_tensor = data_dict['atom_features']

        self.e_features_tensor = data_dict['edge_features']
        self.coordinates = data_dict['coordinates'][:, :3].float()
        if 'conformations' in self.return_types or 'complete_graph_random_conformer' in self.return_types:
            self.conformations = data_dict['coordinates'].float()
            self.conformer_categorical = torch.distributions.Categorical(logits=torch.ones(num_conformers))
        self.edge_indices = data_dict['edge_indices']

        self.meta_dict = {k: data_dict[k] for k in ('chembl_ids', 'edge_slices', 'atom_slices', 'n_atoms')}

        self.atom_padding_indices = torch.tensor(get_atom_feature_dims(), dtype=torch.long, device=device)[None, :]
        self.bond_padding_indices = torch.tensor(get_bond_feature_dims(), dtype=torch.long, device=device)[None, :]

        self.dgl_graphs = {}
        self.pairwise = {}  # for memoization
        self.complete_graphs = {}
        self.mol_complete_graphs = {}
        self.conformer_graphs = {}
        self.pairwise_distances = {}

        self.avg_degree = data_dict['avg_degree']
        # indices of the tasks that should be retrieved
        if 'targets' in self.return_types:
            self.targets = data_dict[target_tasks[0]]
            self.targets_mean = self.targets.mean(dim=0)
            self.targets_std = self.targets.std(dim=0)
            if self.normalize:
                self.targets = ((self.targets - self.targets_mean) / self.targets_std)
            self.targets_mean = self.targets_mean.to(device)
            self.targets_std = self.targets_std.to(device)

    def __len__(self):
        return len(self.meta_dict['chembl_ids'])

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
        e_start = self.meta_dict['edge_slices'][idx].item()
        e_end = self.meta_dict['edge_slices'][idx + 1].item()
        start = self.meta_dict['atom_slices'][idx].item()
        n_atoms = self.meta_dict['n_atoms'][idx].item()

        for return_type in self.return_types:
            data.append(self.data_by_type(idx, return_type, e_start, e_end, start, n_atoms))
        return tuple(data)

    def get_pairwise(self, n_atoms):
        if n_atoms in self.pairwise:
            src, dst = self.pairwise[n_atoms]
            return src.to(self.device), dst.to(self.device)
        else:
            arange = torch.arange(n_atoms, device=self.device)
            src = torch.repeat_interleave(arange, n_atoms - 1)
            dst = torch.cat([torch.cat([arange[:idx], arange[idx + 1:]]) for idx in range(n_atoms)])  # no self loops
            self.pairwise[n_atoms] = (src.to('cpu'), dst.to('cpu'))
            return src, dst

    def get_graph(self, idx, e_start, e_end, n_atoms, start):
        if idx in self.dgl_graphs:
            return self.dgl_graphs[idx].to(self.device)
        else:
            edge_indices = self.edge_indices[:, e_start: e_end]
            g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms, device=self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            self.dgl_graphs[idx] = g.to('cpu')
            return g

    def get_complete_graph(self, idx, n_atoms, start):
        if idx in self.complete_graphs:
            return self.complete_graphs[idx].to(self.device)
        else:
            src, dst = self.get_pairwise(n_atoms)
            g = dgl.graph((src, dst), device=self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2, dim=-1).unsqueeze(
                -1).detach()
            self.complete_graphs[idx] = g.to('cpu')
            return g

    def get_mol_complete_graph(self, idx, e_start, e_end, n_atoms, start):
        if idx in self.mol_complete_graphs:
            return self.mol_complete_graphs[idx].to(self.device)
        else:
            edge_indices = self.edge_indices[:, e_start: e_end]
            src, dst = self.get_pairwise(n_atoms)
            g = dgl.heterograph({('atom', 'bond', 'atom'): (edge_indices[0], edge_indices[1]),
                                 ('atom', 'complete', 'atom'): (src, dst)}, device=self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            self.mol_complete_graphs[idx] = g
            return g

    def data_by_type(self, idx, return_type, e_start, e_end, start, n_atoms):
        if return_type == 'conformations':
            if idx in self.conformer_graphs:
                return self.conformer_graphs[idx].to(self.device)
            else:
                conformer_coords = self.conformations[start: start + n_atoms].to(self.device)
                conformer_graphs = [self.get_complete_graph(idx, n_atoms, start)]
                for i in range(1, self.num_conformers):
                    g = copy.deepcopy(conformer_graphs[0])
                    coords = conformer_coords[:, i * 3:(i + 1) * 3]
                    if torch.equal(coords,conformer_graphs[0].ndata['x']): # add noise to the conformer if it is the same as the first one
                        coords += torch.randn_like(coords, device=self.device) * 0.05
                    g.ndata['x'] = coords
                    g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2,
                                              dim=-1).unsqueeze(-1)
                    conformer_graphs.append(g)
                conformer_graphs = dgl.batch(conformer_graphs)
                self.conformer_graphs[idx] = conformer_graphs.to('cpu')
                return conformer_graphs
        elif return_type == 'dgl_graph':
            g = self.get_graph(idx, e_start, e_end, n_atoms, start)
            return g
        elif return_type == 'complete_graph':  # complete graph without self loops
            g = self.get_complete_graph(idx, n_atoms, start)

            # set edge features with padding for virtual edges
            bond_features = self.e_features_tensor[e_start: e_end].to(self.device)
            e_features = self.bond_padding_indices.expand(n_atoms * n_atoms, -1)
            edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
            bond_indices = edge_indices[0] * n_atoms + edge_indices[1]
            # overwrite the bond features
            e_features = e_features.scatter(dim=0, index=bond_indices[:, None].expand(-1, bond_features.shape[1]),
                                            src=bond_features)
            src, dst = self.get_pairwise(n_atoms)
            g.edata['feat'] = e_features[src * n_atoms + dst]
            return g
        elif return_type == 'complete_graph3d':
            g = self.get_complete_graph(idx, n_atoms, start)
            return g
        elif return_type == 'complete_graph_random_conformer':
            g = self.get_complete_graph(idx, n_atoms, start)
            m = self.conformer_categorical.sample()
            g.ndata['x'] = self.conformations[start: start + n_atoms, m * 3:(m + 1) * 3].to(self.device)
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2, dim=-1).unsqueeze(
                -1)
            return g
        elif return_type == 'mol_complete_graph':
            g = self.get_mol_complete_graph(idx, e_start, e_end, n_atoms, start)
            g.edges['bond'].data['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g
        elif return_type == 'se3Transformer_graph' or return_type == 'se3Transformer_graph3d':
            g = self.get_graph(idx, e_start, e_end, n_atoms, start)
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2, dim=-1).unsqueeze(
                -1)
            if self.e_features_tensor != None and return_type == 'se3Transformer_graph':
                g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g
        elif return_type == 'pairwise_indices':
            src, dst = self.get_pairwise(n_atoms)
            return torch.stack([src, dst], dim=0)
        elif return_type == 'pairwise_distances':
            if idx in self.pairwise_distances:
                return self.pairwise_distances[idx].to(self.device)
            else:
                src, dst = self.get_pairwise(n_atoms)
                coords = self.coordinates[start: start + n_atoms].to(self.device)
                distances = torch.norm(coords[src] - coords[dst], p=2, dim=-1).unsqueeze(-1).detach()
                self.pairwise_distances[idx] = distances.to('cpu')
                return distances
        elif return_type == 'raw_features':
            return self.features_tensor[start: start + n_atoms]
        elif return_type == 'coordinates':
            return self.coordinates[start: start + n_atoms]
        elif return_type == 'targets':
            return self.targets[idx]
        elif return_type == 'edge_indices':
            return self.meta_dict['edge_indices'][:, e_start: e_end]
        elif return_type == 'smiles':
            return self.meta_dict['smiles'][idx]
        else:
            raise Exception(f'return type not supported: ', return_type)

    def process(self):
        print('processing data from ({}) and saving it to ({})'.format(self.root,
                                                                       os.path.join(self.root, 'processed')))
        chembl_ids = os.listdir(os.path.join(self.root, 'structures'))

        targets = {'DFT:ATOMIC_ENERGY': [], 'DFT:TOTAL_ENERGY': [], 'DFT:HOMO_ENERGY': []}
        atom_slices = [0]
        edge_slices = [0]
        all_atom_features = []
        all_edge_features = []
        edge_indices = []  # edges of each molecule in coo format
        total_atoms = 0
        total_edges = 0
        n_atoms_list = []
        coordinates = torch.tensor([])
        avg_degree = 0  # average degree in the dataset
        for mol_idx, chembl_id in tqdm(enumerate(chembl_ids)):
            mol_path = os.path.join(self.root, 'structures', chembl_id)
            sdf_names = os.listdir(mol_path)
            conformers = []
            for conf_idx, sdf_name in enumerate(sdf_names):
                sdf_path = os.path.join(mol_path, sdf_name)
                suppl = Chem.SDMolSupplier(sdf_path)
                mol = next(iter(suppl))
                c = next(iter(mol.GetConformers()))
                conformers.append(torch.tensor(c.GetPositions()))
                if conf_idx == 0:
                    n_atoms = len(mol.GetAtoms())
                    n_atoms_list.append(n_atoms)
                    atom_features_list = []
                    for atom in mol.GetAtoms():
                        atom_features_list.append(atom_to_feature_vector(atom))
                    all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))

                    edges_list = []
                    edge_features_list = []
                    for bond in mol.GetBonds():
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        edge_feature = bond_to_feature_vector(bond)

                        # add edges in both directions
                        edges_list.append((i, j))
                        edge_features_list.append(edge_feature)
                        edges_list.append((j, i))
                        edge_features_list.append(edge_feature)
                    # Graph connectivity in COO format with shape [2, num_edges]
                    edge_index = torch.tensor(edges_list, dtype=torch.long).T
                    edge_features = torch.tensor(edge_features_list, dtype=torch.long)

                    avg_degree += (len(edges_list) / 2) / n_atoms

                    # get all 19 attributes that should be predicted, so we drop the first two entries (name and smiles)
                    targets['DFT:HOMO_ENERGY'].append(float(mol.GetProp('DFT:HOMO_ENERGY')))
                    targets['DFT:TOTAL_ENERGY'].append(float(mol.GetProp('DFT:TOTAL_ENERGY')))
                    targets['DFT:ATOMIC_ENERGY'].append(float(mol.GetProp('DFT:ATOMIC_ENERGY')))
                    edge_indices.append(edge_index)
                    all_edge_features.append(edge_features)

                    total_edges += len(edges_list)
                    total_atoms += n_atoms
                    edge_slices.append(total_edges)
                    atom_slices.append(total_atoms)
            if len(conformers) < 3:  # if there are less than 10 conformers we add the first one a few times
                conformers.extend([conformers[0]] * (3 - len(conformers)))

            coordinates = torch.cat([coordinates, torch.cat(conformers, dim=1)], dim=0)

        data_dict = {'chembl_ids': chembl_ids,
                     'n_atoms': torch.tensor(n_atoms_list, dtype=torch.long),
                     'atom_slices': torch.tensor(atom_slices, dtype=torch.long),
                     'edge_slices': torch.tensor(edge_slices, dtype=torch.long),
                     'edge_indices': torch.cat(edge_indices, dim=1),
                     'atom_features': torch.cat(all_atom_features, dim=0),
                     'edge_features': torch.cat(all_edge_features, dim=0),
                     'coordinates': coordinates,
                     'targets': targets,
                     'avg_degree': avg_degree / len(chembl_ids)
                     }
        for key, value in targets.items():
            targets[key] = torch.tensor(value)[:, None]
        data_dict.update(targets)

        if not os.path.exists(os.path.join(self.root, 'processed')):
            os.mkdir(os.path.join(self.root, 'processed'))
        torch.save(data_dict, os.path.join(self.root, 'processed', self.processed_file))
