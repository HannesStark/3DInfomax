import os

import torch
import dgl
import torch_geometric
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector, get_atom_feature_dims, \
    get_bond_feature_dims
from rdkit import Chem
from rdkit.Chem import rdDistGeom, AllChem
from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs, ETDG
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from scipy.constants import physical_constants

from commons.spherical_encoding import dist_emb

hartree2eV = physical_constants['hartree-electron volt relationship'][0]


class QM9DatasetRDKITConformers(Dataset):
    """The QM9 Dataset. It loads the specified types of data into memory. The processed data is saved in eV units.

    The targets are:
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    not predicted by dimenet, spherical message passing, E(n) equivariant graph neural networks:
    +--------+----------------------------------+-----------------------------------------------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Parameters
    ----------
    return_types: list
        A list with which types of data should be loaded and returened by getitems. Possible options are
        ['dgl_graph', 'complete_graph', 'raw_features', 'coordinates', 'mol_id', 'targets', 'one_hot_bond_types', 'edge_indices', 'smiles', 'atomic_number_long']
        and the default is ['dgl_graph', 'targets']
    features: list
       A list specifying which features should be included in the returned graphs or raw features
       options are ['atom_one_hot', 'atomic_number_long', 'hybridizations', 'is_aromatic', 'constant_ones']
       and default is all except constant ones
    target_tasks: list
        A list specifying which targets should be included in the returend targets, if targets are returned.
        The targets are returned in eV units and saved as eV units in the processed data.
        options are ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']
        and default is ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        which is the stuff that is commonly predicted by papers like DimeNet, Equivariant GNNs, Spherical message passing
        The returned targets will be in the order specified by this list
    normalize: bool
        Whether or not the target (if they should be returned) are normalized to 0 mean and std 1
    prefetch_graphs: bool
        Whether or not to load the dgl graphs into memory. This takes a bit more memory and the upfront computation but
        the graph creation does not have to be done during training which is nice because it takes a long time and can
        slow down training
    """

    def __init__(self, return_types: list = None,
                 target_tasks: list = None,
                 normalize: bool = True, device='cuda:0', dist_embedding: bool = False, num_radial: int = 6,
                 transform=None, **kwargs):
        self.qm9_directory = 'dataset/QM9'
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

        self.return_types: list = return_types

        if target_tasks == None or target_tasks == []:  # set default
            self.target_tasks = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        else:
            self.target_tasks: list = target_tasks
        for target_task in self.target_tasks:
            assert target_task in self.unit_conversion.keys()

        # load the data and get normalization values
        if not os.path.exists(os.path.join(self.qm9_directory, 'processed_rdkit_conformers_120k', self.processed_file)):
            self.process()
        data_dict = torch.load(os.path.join(self.qm9_directory, 'processed_rdkit_conformers_120k', self.processed_file))

        self.features_tensor = data_dict['atom_features']

        self.e_features_tensor = data_dict['edge_features']
        self.coordinates = data_dict['coordinates']
        self.edge_indices = data_dict['edge_indices']

        self.meta_dict = {k: data_dict[k] for k in ('mol_id', 'edge_slices', 'atom_slices', 'n_atoms')}

        if 'san_graph' in self.return_types or 'positional_encoding' in self.return_types:
            self.eig_vals = data_dict['eig_vals']
            self.eig_vecs = data_dict['eig_vecs']

        if 'smiles' in self.return_types:
            self.smiles = pd.read_csv(os.path.join(self.qm9_directory, self.raw_qm9_file))['smiles']

        self.dgl_graphs = {}
        self.pairwise = {}  # for memoization
        self.complete_graphs = {}
        self.mol_complete_graphs = {}
        self.pairwise_distances = {}

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
        if return_type == 'dgl_graph':
            return self.get_graph(idx, e_start, e_end, n_atoms, start)
        elif return_type == 'complete_graph':  # complete graph without self loops
            g = self.get_complete_graph(idx, n_atoms, start)

            # set edge features with padding for virtual edges
            bond_features = self.e_features_tensor[e_start: e_end].to(self.device)
            # TODO: replace with -1 padding
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
            g = self.get_complete_graph(idx, n_atoms, start)
            if self.dist_embedding:
                g.edata['d_rbf'] = self.dist_embedder(g.edata['feat']).to(self.device)
            return g
        if return_type == 'mol_complete_graph':
            g = self.get_mol_complete_graph(idx, e_start, e_end, n_atoms, start)
            if self.e_features_tensor != None:
                g.edges['bond'].data['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g
        elif return_type == 'san_graph':
            g = self.get_complete_graph(idx, n_atoms, start).to(self.device)
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
            g = self.get_graph(idx, e_start, e_end, n_atoms, start)
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2, dim=-1).unsqueeze(
                -1)
            if self.e_features_tensor != None and return_type == 'se3Transformer_graph':
                g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g
        elif return_type == 'padded_e_features':
            bond_features = self.e_features_tensor[e_start: e_end].to(self.device)
            # TODO: replace with -1 padding
            e_features = self.bond_padding_indices.expand(n_atoms * n_atoms, -1)
            edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
            bond_indices = edge_indices[0] * n_atoms + edge_indices[1]
            # overwrite the bond features
            return e_features.scatter(dim=0, index=bond_indices[:, None].expand(-1, bond_features.shape[1]),
                                      src=bond_features)
        elif return_type == 'pytorch_geometric_smp_graph':
            R_i = self.coordinates[start: start + n_atoms].to(self.device)
            z_i = self.features_tensor[start: start + n_atoms].to(self.device)
            return torch_geometric.data.Data(pos=R_i, z=z_i)
        elif return_type == 'pytorch_geometric_graph':
            edge_features = self.e_features_tensor[e_start: e_end].to(self.device)
            edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
            R_i = self.coordinates[start: start + n_atoms].to(self.device)
            z_i = self.features_tensor[start: start + n_atoms].to(self.device)
            return torch_geometric.data.Data(pos=R_i, z=z_i, edge_attr=edge_features, edge_index=edge_indices)
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
        print('processing data from ({}) and saving it to ({})'.format(self.qm9_directory,
                                                                       os.path.join(self.qm9_directory,
                                                                                    'processed_rdkit_conformers_120k')))
        # load qm9 data with spatial coordinates
        data_qm9 = dict(np.load(os.path.join(self.qm9_directory, self.raw_spatial_data), allow_pickle=True))
        # Read the QM9 data with SMILES information
        molecules_df = pd.read_csv(os.path.join(self.qm9_directory, self.raw_qm9_file))

        atom_slices = [0]
        edge_slices = [0]
        mol_id = []
        n_atoms_list = []
        all_atom_features = []
        all_edge_features = []
        coordinates = []
        edge_indices = []  # edges of each molecule in coo format
        targets = []  # the 19 properties that should be predicted for the QM9 dataset
        total_atoms = 0
        total_edges = 0
        avg_degree = 0  # average degree in the dataset
        # go through all molecules in the npz file

        for mol_idx, n_atoms in tqdm(enumerate(data_qm9['N'])):
            # get the molecule using the smiles representation from the csv file
            mol = Chem.MolFromSmiles(molecules_df['smiles'][data_qm9['id'][mol_idx]])
            # add hydrogen bonds to molecule because they are not in the smiles representation
            mol = Chem.AddHs(mol)

            try:
                ps = AllChem.ETKDGv2()
                ps.useRandomCoords = True
                AllChem.EmbedMolecule(mol, ps)
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
                conf = mol.GetConformer()
                coordinates.append(torch.tensor(conf.GetPositions(), dtype=torch.float))
            except:
                print(molecules_df['smiles'][data_qm9['id'][mol_idx]])
                continue

            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))

            # adj = GetAdjacencyMatrix(mol, useBO=False, force=True)
            # max_freqs = 10
            # adj = torch.tensor(adj).float()
            # D = torch.diag(adj.sum(dim=0))
            # L = D - adj
            # N = adj.sum(dim=0) ** -0.5
            # L_sym = torch.eye(n_atoms) - N * L * N
            # eig_vals, eig_vecs =torch.linalg.eigh(L_sym, eigenvectors=True)
            # idx = eig_vals.argsort()[0: max_freqs]  # Keep up to the maximum desired number of frequencies
            # eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]
            #
            ## Sort, normalize and pad EigenVectors
            # eig_vecs = eig_vecs[:, eig_vals.argsort()]  # increasing order
            # eig_vecs = F.normalize(eig_vecs, p=2, dim=1, eps=1e-12, out=None)
            # if n_atoms < max_freqs:
            #    eig_vecs = F.pad(eig_vecs, (0, max_freqs - n_atoms), value=float('nan'))
            #    eig_vals = F.pad(eig_vals, (0, max_freqs - n_atoms), value=float('nan'))

            # total_eigvecs.append(eig_vecs)
            # total_eigvals.append(eig_vals.unsqueeze(0))

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
            target = torch.tensor(molecules_df.iloc[data_qm9['id'][mol_idx]][2:], dtype=torch.float)
            targets.append(target)
            edge_indices.append(edge_index)
            all_edge_features.append(edge_features)

            total_edges += len(edges_list)
            total_atoms += n_atoms
            edge_slices.append(total_edges)
            atom_slices.append(total_atoms)
            mol_id.append(data_qm9['id'][mol_idx])
            n_atoms_list.append(n_atoms)

        # convert targets to eV units
        targets = torch.stack(targets) * torch.tensor(list(self.unit_conversion.values()))[None, :]
        data_dict = {'mol_id': torch.tensor(mol_id, dtype=torch.long),
                     'n_atoms': torch.tensor(n_atoms_list, dtype=torch.long),
                     'atom_slices': torch.tensor(atom_slices, dtype=torch.long),
                     'edge_slices': torch.tensor(edge_slices, dtype=torch.long),
                     'edge_indices': torch.cat(edge_indices, dim=1),
                     'atom_features': torch.cat(all_atom_features, dim=0),
                     'edge_features': torch.cat(all_edge_features, dim=0),
                     'coordinates': torch.cat(coordinates, dim=0),
                     'targets': targets,
                     'avg_degree': avg_degree / len(data_qm9['id'])
                     }

        if not os.path.exists(os.path.join(self.qm9_directory, 'processed_rdkit_conformers_120k')):
            os.mkdir(os.path.join(self.qm9_directory, 'processed_rdkit_conformers_120k'))
        torch.save(data_dict, os.path.join(self.qm9_directory, 'processed_rdkit_conformers_120k', self.processed_file))
