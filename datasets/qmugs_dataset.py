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

    def __init__(self, return_types: list = None,
                 target_tasks: list = None,
                 normalize: bool = True, device='cuda:0', dist_embedding: bool = False, num_radial: int = 6,
                 prefetch_graphs=True, transform=None, **kwargs):
        self.return_type_options = ['mol_graph', 'complete_graph', 'mol_graph3d', 'complete_graph3d', 'san_graph',
                                    'mol_complete_graph', 'se3Transformer_graph', 'se3Transformer_graph3d',
                                    'pairwise_distances', 'pairwise_distances_squared', 'pairwise_indices',
                                    'raw_features', 'coordinates', 'dist_embedding', 'mol_id', 'targets',
                                    'one_hot_bond_types', 'edge_indices', 'smiles', 'atomic_number_long', 'n_atoms',
                                    'positional_encoding', 'constant_ones', 'pytorch_geometric_graph']
        self.root = '../../QMugs'
        self.processed_file = 'processed.pt'
        self.raw_csv = 'summary.csv'
        self.normalize = normalize
        self.device = device
        self.transform = transform

        self.num_radial = num_radial
        # data in the csv file is in Hartree units.
        self.unit_conversion = {'A': 1.0,
                                'B': 1.0,
                                'C': 1.0,
                                'mu': 1.0,
                                'alpha': 1.0,
                                'homo': hartree2eV,
                                'lumo': hartree2eV,
                                'gap': hartree2eV,
                                'r2': 1.0,
                                'zpve': hartree2eV,
                                'u0': hartree2eV,
                                'u298': hartree2eV,
                                'h298': hartree2eV,
                                'g298': hartree2eV,
                                'cv': 1.0,
                                'u0_atom': hartree2eV,
                                'u298_atom': hartree2eV,
                                'h298_atom': hartree2eV,
                                'g298_atom': hartree2eV}

        if return_types == None:  # set default
            self.return_types: list = ['mol_graph', 'targets']
        else:
            self.return_types: list = return_types
        for return_type in self.return_types:
            if not return_type in self.return_type_options: raise Exception(f'return_type not supported: {return_type}')

        if target_tasks == None or target_tasks == []:  # set default
            self.target_tasks = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        else:
            self.target_tasks: list = target_tasks
        for target_task in self.target_tasks:
            assert target_task in self.unit_conversion.keys()

        # load the data and get normalization values
        if not os.path.exists(os.path.join(self.root, 'processed', self.processed_file)):
            self.process()
        data_dict = torch.load(os.path.join(self.root, 'processed', self.processed_file))

        self.features_tensor = data_dict['atom_features']

        self.e_features_tensor = data_dict['edge_features']
        self.coordinates = data_dict['coordinates'][:, :3]
        self.edge_indices = data_dict['edge_indices']

        self.meta_dict = {k: data_dict[k] for k in ('chembl_ids','edge_slices', 'atom_slices', 'n_atoms')}

        self.atom_padding_indices = torch.tensor(get_atom_feature_dims(), dtype=torch.long, device=device)[None, :]
        self.bond_padding_indices = torch.tensor(get_bond_feature_dims(), dtype=torch.long, device=device)[None, :]



        self.prefetch_graphs = prefetch_graphs
        if self.prefetch_graphs and any(return_type in self.return_types for return_type in
                                        ['mol_graph', 'mol_graph3d', 'se3Transformer_graph', 'se3Transformer_graph3d']):
            print(
                'Load molecular graphs into memory (set prefetch_graphs to False to load them on the fly => slower training)')
            self.mol_graphs = []
            for idx, n_atoms in tqdm(enumerate(self.meta_dict['n_atoms'])):
                e_start = self.meta_dict['edge_slices'][idx]
                e_end = self.meta_dict['edge_slices'][idx + 1]
                edge_indices = self.edge_indices[:, e_start: e_end]
                self.mol_graphs.append(dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms))
        self.pairwise = {}  # for memoization
        if self.prefetch_graphs and (
                'complete_graph' in self.return_types or 'complete_graph3d' in self.return_types or 'san_graph' in self.return_types):
            print(
                'Load complete graphs into memory (set prefetch_graphs to False to load them on the fly => slower training)')
            self.complete_graphs = []
            for idx, n_atoms in tqdm(enumerate(self.meta_dict['n_atoms'])):
                src, dst = self.get_pairwise(n_atoms)
                self.complete_graphs.append(dgl.graph((src, dst)))
        if self.prefetch_graphs and (
                'mol_complete_graph' in self.return_types or 'mol_complete_graph3d' in self.return_types):
            print(
                'Load mol_complete_graph graphs into memory (set prefetch_graphs to False to load them on the fly => slower training)')
            self.mol_complete_graphs = []
            for idx, n_atoms in tqdm(enumerate(self.meta_dict['n_atoms'])):
                src, dst = self.get_pairwise(n_atoms)
                self.mol_complete_graphs.append(
                    dgl.heterograph({('atom', 'bond', 'atom'): (src, dst), ('atom', 'complete', 'atom'): (src, dst)}))
        print('Finish loading data into memory')

        self.avg_degree = data_dict['avg_degree']
        # indices of the tasks that should be retrieved
        self.task_indices = torch.tensor([list(self.unit_conversion.keys()).index(task) for task in self.target_tasks])
        # select targets in the order specified by the target_tasks argument


    def get_pairwise(self, n_atoms):
        if n_atoms in self.pairwise:
            return self.pairwise[n_atoms]
        else:
            src = torch.repeat_interleave(torch.arange(n_atoms), n_atoms - 1)
            dst = torch.cat([torch.cat([torch.arange(n_atoms)[:idx], torch.arange(n_atoms)[idx + 1:]]) for idx in
                             range(n_atoms)])  # without self loops
            self.pairwise[n_atoms] = (src, dst)
            return src, dst

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
        e_start = self.meta_dict['edge_slices'][idx]
        e_end = self.meta_dict['edge_slices'][idx + 1]
        start = self.meta_dict['atom_slices'][idx]
        n_atoms = self.meta_dict['n_atoms'][idx]

        for return_type in self.return_types:
            data.append(self.data_by_type(idx, return_type, e_start, e_end, start, n_atoms))
        return tuple(data)

    def get_graph(self, idx, e_start, e_end, n_atoms):
        if self.prefetch_graphs:
            g = self.mol_graphs[idx]
        else:
            edge_indices = self.edge_indices[:, e_start: e_end]
            g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms)
        return g

    def get_complete_graph(self, idx, n_atoms):
        if self.prefetch_graphs:
            g = self.complete_graphs[idx]
        else:
            src, dst = self.get_pairwise(n_atoms)
            g = dgl.graph((src, dst))
        return g

    def get_mol_complete_graph(self, idx, e_start, e_end, n_atoms):
        if self.prefetch_graphs:
            g = self.mol_complete_graphs[idx]
        else:
            edge_indices = self.edge_indices[:, e_start: e_end]
            src, dst = self.get_pairwise(n_atoms)
            g = dgl.heterograph({('atom', 'bond', 'atom'): (edge_indices[0], edge_indices[1]),
                                 ('atom', 'complete', 'atom'): (src, dst)})
        return g

    def data_by_type(self, idx, return_type, e_start, e_end, start, n_atoms):
        if return_type == 'mol_graph':
            g = self.get_graph(idx, e_start, e_end, n_atoms).to(self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g
        elif return_type == 'mol_graph3d':
            g = self.get_graph(idx, e_start, e_end, n_atoms).to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            return g
        elif return_type == 'complete_graph':  # complete graph without self loops
            g = self.get_complete_graph(idx, n_atoms).to(self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2, dim=-1).unsqueeze(
                -1)

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
            if self.dist_embedding:
                g.edata['d_rbf'] = self.dist_embedder(g.edata['feat']).to(self.device)
            return g
        elif return_type == 'complete_graph3d':
            g = self.get_complete_graph(idx, n_atoms).to(self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2, dim=-1).unsqueeze(
                -1)
            if self.dist_embedding:
                g.edata['d_rbf'] = self.dist_embedder(g.edata['feat']).to(self.device)
            return g
        if return_type == 'mol_complete_graph':
            g = self.get_mol_complete_graph(idx, e_start, e_end, n_atoms).to(self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            if self.e_features_tensor != None:
                g.edges['bond'].data['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g
        if return_type == 'san_graph':
            g = self.get_complete_graph(idx, n_atoms).to(self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device).float()
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            eig_vals = self.eig_vals[idx].to(self.device)
            sign_flip = torch.rand(eig_vals.shape[0], device=self.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eig_vecs = self.eig_vecs[start: start + n_atoms].to(self.device) * sign_flip.unsqueeze(0)
            eig_vals = eig_vals.unsqueeze(0).repeat(n_atoms, 1)
            g.ndata['pos_enc'] = torch.stack([eig_vals, eig_vecs], dim=-1)
            if self.e_features_tensor != None:
                e_features = self.e_features_tensor[e_start: e_end].to(self.device).float()
                g.edata['feat'] = torch.zeros(g.number_of_edges(), e_features.shape[1], dtype=torch.float32,
                                              device=self.device)
                g.edata['real'] = torch.zeros(g.number_of_edges(), dtype=torch.long, device=self.device)
                edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
                g.edges[edge_indices[0], edge_indices[1]].data['feat'] = e_features
                g.edges[edge_indices[0], edge_indices[1]].data['real'] = torch.ones(e_features.shape[0],
                                                                                    dtype=torch.long,
                                                                                    device=self.device)  # This indicates real edges
            return g
        elif return_type == 'se3Transformer_graph' or return_type == 'se3Transformer_graph3d':
            g = self.get_graph(idx, e_start, e_end, n_atoms).to(self.device)
            x = self.coordinates[start: start + n_atoms].to(self.device)
            if self.transform:
                x = self.transform(x)
            g.ndata['x'] = x
            edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
            g.edata['d'] = x[edge_indices[0]] - x[edge_indices[1]]
            if self.e_features_tensor != None and return_type == 'se3Transformer_graph':
                g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g

        elif return_type == 'padded_e_features':
            bond_features = self.e_features_tensor[e_start: e_end].to(self.device)
            e_features = self.bond_padding_indices.expand(n_atoms * n_atoms, -1)
            edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
            bond_indices = edge_indices[0] * n_atoms + edge_indices[1]
            # overwrite the bond features
            return e_features.scatter(dim=0, index=bond_indices[:, None].expand(-1, bond_features.shape[1]),
                                      src=bond_features)
        elif return_type == 'pytorch_geometric_graph':
            R_i = self.coordinates[start: start + n_atoms].to(self.device)
            z_i = self.features_tensor[start: start + n_atoms].to(self.device)
            return torch_geometric.data.Data(pos=R_i, z=z_i)
        elif return_type == 'pairwise_indices':
            src, dst = self.get_pairwise(n_atoms)
            return torch.stack([src, dst], dim=0).to(self.device)
        elif return_type == 'raw_features':
            return self.features_tensor[start: start + n_atoms].to(self.device)
        elif return_type == 'constant_ones':
            return torch.ones_like(self.features_tensor[start: start + n_atoms], device=self.device)
        elif return_type == 'n_atoms':
            return self.meta_dict['n_atoms'][n_atoms]
        elif return_type == 'coordinates':
            return self.coordinates[start: start + n_atoms].to(self.device)
        elif return_type == 'positional_encoding':
            eig_vals = self.eig_vals[idx].to(self.device)
            sign_flip = torch.rand(eig_vals.shape[0], device=self.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eig_vecs = self.eig_vecs[start: start + n_atoms].to(self.device) * sign_flip.unsqueeze(0)
            eig_vals = eig_vals.unsqueeze(0).repeat(n_atoms, 1)
            return torch.stack([eig_vals, eig_vecs], dim=-1)
        elif return_type == 'mol_id':
            return self.meta_dict['mol_id'][idx]
        elif return_type == 'targets':
            return self.targets[idx]
        elif return_type == 'edge_indices':
            return self.meta_dict['edge_indices'][:, e_start: e_end]
        elif return_type == 'smiles':
            return self.smiles[self.meta_dict['mol_id'][idx]]
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

            coordinates = torch.cat([coordinates, torch.cat(conformers,dim=1)], dim=0)

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
