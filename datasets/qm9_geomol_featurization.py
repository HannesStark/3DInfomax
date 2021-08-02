import os

import torch
import dgl
import torch_geometric
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector, get_atom_feature_dims, \
    get_bond_feature_dims
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch_scatter import scatter
from tqdm import tqdm
import torch.nn.functional as F
from scipy.constants import physical_constants

from commons.spherical_encoding import dist_emb

hartree2eV = physical_constants['hartree-electron volt relationship'][0]
dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
         'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
         'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
         'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}

class QM9GeomolFeaturization(Dataset):
    def __init__(self, return_types: list = None,
                 target_tasks: list = None,
                 normalize: bool = True, device='cuda:0', dist_embedding: bool = False, num_radial: int = 6,
                 prefetch_graphs=True, transform=None, **kwargs):
        self.return_type_options = ['dgl_graph', 'complete_graph', 'dgl_graph3d', 'complete_graph3d', 'san_graph',
                                    'mol_complete_graph', 'se3Transformer_graph', 'se3Transformer_graph3d',
                                    'pairwise_distances', 'pairwise_distances_squared', 'pairwise_indices',
                                    'raw_features', 'coordinates', 'dist_embedding', 'mol_id', 'targets',
                                    'one_hot_bond_types', 'edge_indices', 'smiles', 'atomic_number_long', 'n_atoms',
                                    'positional_encoding', 'constant_ones', 'pytorch_geometric_graph']
        self.qm9_directory = 'dataset/QM9Geomol'
        self.processed_file = 'qm9_processed.pt'
        self.distances_file = 'qm9_distances.pt'
        self.raw_qm9_file = 'qm9.csv'
        self.raw_spatial_data = 'qm9_eV.npz'
        self.atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
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
            self.return_types: list = ['dgl_graph', 'targets']
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
        if not os.path.exists(os.path.join(self.qm9_directory, 'processed', self.processed_file)):
            self.process()
        data_dict = torch.load(os.path.join(self.qm9_directory, 'processed', self.processed_file))

        self.features_tensor = data_dict['atom_features']

        self.e_features_tensor = data_dict['edge_features']
        self.coordinates = data_dict['coordinates']
        self.edge_indices = data_dict['edge_indices']

        self.meta_dict = {k: data_dict[k] for k in ('mol_id', 'edge_slices', 'atom_slices', 'n_atoms')}

        self.atom_padding_indices = torch.tensor(get_atom_feature_dims(), dtype=torch.long, device=device)[None, :]
        self.bond_padding_indices = torch.tensor(get_bond_feature_dims(), dtype=torch.long, device=device)[None, :]

        if 'san_graph' in self.return_types or 'positional_encoding' in self.return_types:
            self.eig_vals = data_dict['eig_vals']
            self.eig_vecs = data_dict['eig_vecs']

        if 'smiles' in self.return_types:
            self.smiles = pd.read_csv(os.path.join(self.qm9_directory, self.raw_qm9_file))['smiles']

        self.prefetch_graphs = prefetch_graphs
        if self.prefetch_graphs and any(return_type in self.return_types for return_type in
                                        ['dgl_graph', 'dgl_graph3d', 'se3Transformer_graph', 'se3Transformer_graph3d']):
            print(
                'Load molecular graphs into memory (set prefetch_graphs to False to load them on the fly => slower training)')
            self.dgl_graphs = []
            for idx, n_atoms in tqdm(enumerate(self.meta_dict['n_atoms'])):
                e_start = self.meta_dict['edge_slices'][idx]
                e_end = self.meta_dict['edge_slices'][idx + 1]
                edge_indices = self.edge_indices[:, e_start: e_end]
                self.dgl_graphs.append(dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms))
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

        self.targets = data_dict['targets'].index_select(dim=1, index=self.task_indices)  # [130831, n_tasks]
        self.targets_mean = self.targets.mean(dim=0)
        self.targets_std = self.targets.std(dim=0)
        if self.normalize:
            self.targets = ((self.targets - self.targets_mean) / self.targets_std)
        self.targets_mean = self.targets_mean.to(device)
        self.targets_std = self.targets_std.to(device)
        # get a tensor that is 1000 for all targets that are energies and 1.0 for all other ones
        self.eV2meV = torch.tensor(
            [1.0 if list(self.unit_conversion.values())[task_index] == 1.0 else 1000 for task_index in
             self.task_indices]).to(self.device)  # [n_tasks]
        self.dist_embedder = dist_emb(num_radial=6).to(device)
        self.dist_embedding = dist_embedding

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
        return len(self.meta_dict['mol_id'])

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
            g = self.dgl_graphs[idx]
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
        if return_type == 'dgl_graph':
            g = self.get_graph(idx, e_start, e_end, n_atoms).to(self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.ndata['x'] = self.coordinates[start: start + n_atoms].to(self.device)
            g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g
        elif return_type == 'dgl_graph3d':
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
            edge_features = self.e_features_tensor[e_start: e_end].to(self.device)
            edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
            R_i = self.coordinates[start: start + n_atoms].to(self.device)
            z_i = self.features_tensor[start: start + n_atoms].to(self.device)
            return torch_geometric.data.Data(pos=R_i, z=z_i, edge_attr=edge_features, edge_index=edge_indices)
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

    def one_k_encoding(self, value, choices):
        """
        Creates a one-hot encoding with an extra category for uncommon values.
        :param value: The value for which the encoding should be one.
        :param choices: A list of possible values.
        :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
                 If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
        """
        encoding = [0] * (len(choices) + 1)
        index = choices.index(value) if value in choices else -1
        encoding[index] = 1

        return encoding

    def featurize_mol(self, mol):
        N = mol.GetNumAtoms()

        type_idx = []
        atomic_number = []
        atom_features = []
        chiral_tag = []
        neighbor_dict = {}
        ring = mol.GetRingInfo()
        for i, atom in enumerate(mol.GetAtoms()):
            type_idx.append(types[atom.GetSymbol()])
            n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(n_ids) > 1:
                neighbor_dict[i] = torch.tensor(n_ids)
            chiral_tag.append(chirality[atom.GetChiralTag()])
            atomic_number.append(atom.GetAtomicNum())
            atom_features.extend([atom.GetAtomicNum(),
                                  1 if atom.GetIsAromatic() else 0])
            atom_features.extend(self.one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(self.one_k_encoding(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2]))
            atom_features.extend(self.one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(self.one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
            atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                  int(ring.IsAtomInRingOfSize(i, 4)),
                                  int(ring.IsAtomInRingOfSize(i, 5)),
                                  int(ring.IsAtomInRingOfSize(i, 6)),
                                  int(ring.IsAtomInRingOfSize(i, 7)),
                                  int(ring.IsAtomInRingOfSize(i, 8))])
            atom_features.extend(self.one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))


        row, col, edge_type, bond_features = [], [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]

            bond_features += 2 * [int(bond.IsInRing()),
                                  int(bond.GetIsConjugated()),
                                  int(bond.GetIsAromatic())]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
        x2 = torch.tensor(atom_features).view(N, -1)
        x = torch.cat([x1.to(torch.float), x2], dim=-1)

        return x, edge_index, edge_attr

    def process(self):

        print('processing data from ({}) and saving it to ({})'.format(self.qm9_directory,
                                                                       os.path.join(self.qm9_directory, 'processed')))

        # load qm9 data with spatial coordinates
        data_qm9 = dict(np.load(os.path.join(self.qm9_directory, self.raw_spatial_data), allow_pickle=True))
        coordinates = torch.tensor(data_qm9['R'], dtype=torch.float)
        # Read the QM9 data with SMILES information
        molecules_df = pd.read_csv(os.path.join(self.qm9_directory, self.raw_qm9_file))

        atom_slices = [0]
        edge_slices = [0]
        total_eigvecs = []
        total_eigvals = []
        all_atom_features = []
        all_edge_features = []
        edge_indices = []  # edges of each molecule in coo format
        targets = []  # the 19 properties that should be predicted for the QM9 dataset
        total_atoms = 0
        total_edges = 0
        avg_degree = 0  # average degree in the dataset
        # go through all molecules in the npz file
        for mol_idx, n_atoms in tqdm(enumerate(data_qm9['N'])):
            # get the molecule using the smiles representation from the csv file
            smiles = molecules_df['smiles'][data_qm9['id'][mol_idx]]
            mol = Chem.MolFromSmiles(smiles)
            # add hydrogen bonds to molecule because they are not in the smiles representation
            mol = Chem.AddHs(mol)

            x, edge_index, edge_features = self.featurize_mol(mol)
            assert len(x) == n_atoms
            all_atom_features.append(x)

            adj = GetAdjacencyMatrix(mol, useBO=False, force=True)
            max_freqs = 10
            adj = torch.tensor(adj).float()
            D = torch.diag(adj.sum(dim=0))
            L = D - adj
            N = adj.sum(dim=0) ** -0.5
            L_sym = torch.eye(n_atoms) - N * L * N
            eig_vals, eig_vecs =torch.linalg.eigh(L_sym)
            idx = eig_vals.argsort()[0: max_freqs]  # Keep up to the maximum desired number of frequencies
            eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]

            # Sort, normalize and pad EigenVectors
            eig_vecs = eig_vecs[:, eig_vals.argsort()]  # increasing order
            eig_vecs = F.normalize(eig_vecs, p=2, dim=1, eps=1e-12, out=None)
            if n_atoms < max_freqs:
                eig_vecs = F.pad(eig_vecs, (0, max_freqs - n_atoms), value=float('nan'))
                eig_vals = F.pad(eig_vals, (0, max_freqs - n_atoms), value=float('nan'))

            total_eigvecs.append(eig_vecs)
            total_eigvals.append(eig_vals.unsqueeze(0))


            avg_degree += (edge_index.shape[1] / 2) / n_atoms

            # get all 19 attributes that should be predicted, so we drop the first two entries (name and smiles)
            target = torch.tensor(molecules_df.iloc[data_qm9['id'][mol_idx]][2:], dtype=torch.float)
            targets.append(target)
            edge_indices.append(edge_index)
            all_edge_features.append(edge_features)

            total_edges += edge_index.shape[1]
            total_atoms += n_atoms
            edge_slices.append(total_edges)
            atom_slices.append(total_atoms)

        # convert targets to eV units
        targets = torch.stack(targets) * torch.tensor(list(self.unit_conversion.values()))[None, :]
        data_dict = {'mol_id': data_qm9['id'],
                     'n_atoms': torch.tensor(data_qm9['N'], dtype=torch.long),
                     'atom_slices': torch.tensor(atom_slices, dtype=torch.long),
                     'edge_slices': torch.tensor(edge_slices, dtype=torch.long),
                     'eig_vecs': torch.cat(total_eigvecs).float(),
                     'eig_vals': torch.cat(total_eigvals).float(),
                     'edge_indices': torch.cat(edge_indices, dim=1),
                     'atom_features': torch.cat(all_atom_features, dim=0),
                     'edge_features': torch.cat(all_edge_features, dim=0),
                     'atomic_number_long': torch.tensor(data_qm9['Z'], dtype=torch.long)[:, None],
                     'coordinates': coordinates,
                     'targets': targets,
                     'avg_degree': avg_degree / len(data_qm9['id'])
                     }

        if not os.path.exists(os.path.join(self.qm9_directory, 'processed')):
            os.mkdir(os.path.join(self.qm9_directory, 'processed'))
        torch.save(data_dict, os.path.join(self.qm9_directory, 'processed', self.processed_file))
