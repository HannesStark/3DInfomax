import numpy as np
import copy
import json
import os
import pickle

import torch
import dgl
import torch_geometric
from ogb.utils.features import bond_to_feature_vector, atom_to_feature_vector, get_atom_feature_dims, \
    get_bond_feature_dims
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from scipy.constants import physical_constants

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}


class QM9GeomolFeatDataset(Dataset):

    def __init__(self, return_types: list = None, target_tasks: list = None, normalize: bool = True, device='cuda:0',
                 num_conformers=1, **kwargs):

        self.target_types = ['ensembleenergy', 'ensembleentropy', 'ensemblefreeenergy', 'lowestenergy', 'poplowestpct',
                             'temperature', 'uniqueconfs']
        self.directory = 'dataset/GEOM'
        self.processed_file = 'geomol_geom_qm9_processed.pt'
        self.atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'P': 9, 'B': 10,
                           'Bi': 11, 'Si': 12, 'As': 13, 'Al': 14, 'Hg': 15}
        self.symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15, 'B': 5,
                        'Bi': 83, 'Si': 14, 'As': 33, 'Al': 13, 'Hg': 80}
        self.normalize = normalize
        self.device = device
        self.num_conformers = num_conformers
        self.return_types: list = return_types

        # load the data and get normalization values
        if not os.path.exists(os.path.join(self.directory, 'processed', self.processed_file)):
            self.process()
        print('load pickle')
        data_dict = torch.load(os.path.join(self.directory, 'processed', self.processed_file))
        print('finish loading')

        self.features_tensor = data_dict['atom_features']
        self.e_features_tensor = data_dict['edge_features']

        self.coordinates = data_dict['coordinates'][:, :3]
        if 'conformations' in self.return_types or 'complete_graph_random_conformer' in self.return_types or 'pyg_multiple_conformers' in self.return_types:
            self.conformations = data_dict['coordinates'][:, :self.num_conformers*3]
            self.conformations = torch.stack(self.conformations.split(3, dim=1), dim=1)
            self.conformer_categorical = torch.distributions.Categorical(logits=torch.ones(self.num_conformers))
        self.edge_indices = data_dict['edge_indices']

        self.meta_dict = {k: data_dict[k] for k in (
            'smiles', 'edge_slices', 'atom_slices', 'n_atoms', 'neighbors_list', 'neighbors_slices', 'atoms_with_neighbors_slices', 'neighbors_idx')}

        self.pos_masks = data_dict['pos_masks']
        self.chiral_tags = data_dict['chiral_tags']

        if 'san_graph' in self.return_types:
            self.eig_vals = data_dict['eig_vals']
            self.eig_vecs = data_dict['eig_vecs']

        self.dgl_graphs = {}
        self.pairwise = {}  # for memoization
        self.complete_graphs = {}
        self.mol_complete_graphs = {}
        self.conformer_graphs = {}

        self.avg_degree = data_dict['avg_degree']
        # indices of the tasks that should be retrieved
        # select targets in the order specified by the target_tasks argument
        if 'targets' in self.return_types:
            self.targets = data_dict[target_tasks[0]]
            self.targets_mean = self.targets.mean(dim=0)
            self.targets_std = self.targets.std(dim=0)
            if self.normalize:
                self.targets = ((self.targets - self.targets_mean) / self.targets_std)
            self.targets_mean = self.targets_mean.to(device)
            self.targets_std = self.targets_std.to(device)

    def __len__(self):
        return len(self.meta_dict['smiles'])

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
                    if torch.equal(coords, conformer_graphs[0].ndata[
                        'x']):  # add noise to the conformer if it is the same as the first one
                        coords += torch.randn_like(coords, device=self.device) * 0.05
                    g.ndata['x'] = coords
                    g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2,
                                              dim=-1).unsqueeze(-1)
                    conformer_graphs.append(g)
                conformer_graphs = dgl.batch(conformer_graphs)
                self.conformer_graphs[idx] = conformer_graphs.to('cpu')
                return conformer_graphs
        elif return_type == 'dgl_graph':
            return self.get_graph(idx, e_start, e_end, n_atoms, start)
        elif return_type == 'neighbors':
            slices = self.meta_dict['neighbors_slices'][start: start + n_atoms]
            neighbors = []
            for i, slice in enumerate(slices[:-1]):
                neighbors.append(self.meta_dict['neighbors_list'][slice: slices[i + 1]])
            return neighbors
        elif return_type == 'dict_neighbors':
            return self.meta_dict['neighbor_dicts'][idx]
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
        elif return_type == 'san_graph':
            g = self.get_complete_graph(idx, n_atoms, start)
            eig_vals = self.eig_vals[idx].to(self.device)
            sign_flip = torch.rand(eig_vals.shape[0], device=self.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eig_vecs = self.eig_vecs[start: start + n_atoms].to(self.device) * sign_flip.unsqueeze(0)
            eig_vals = eig_vals.unsqueeze(0).repeat(n_atoms, 1)
            g.ndata['pos_enc'] = torch.stack([eig_vals, eig_vecs], dim=-1)

            e_features = self.e_features_tensor[e_start: e_end].to(self.device)
            g.edata['feat'] = torch.zeros(g.number_of_edges(), e_features.shape[1], dtype=torch.float32,
                                          device=self.device)
            g.edata['real'] = torch.zeros(g.number_of_edges(), dtype=torch.long, device=self.device)
            edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
            g.edges[edge_indices[0], edge_indices[1]].data['feat'] = e_features
            g.edges[edge_indices[0], edge_indices[1]].data['real'] = torch.ones(e_features.shape[0], dtype=torch.long,
                                                                                device=self.device)  # This indicates real edges
            return g
        elif return_type == 'se3Transformer_graph' or return_type == 'se3Transformer_graph3d':
            g = self.get_graph(idx, e_start, e_end, n_atoms, start)
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2, dim=-1).unsqueeze(
                -1)
            if self.e_features_tensor != None and return_type == 'se3Transformer_graph':
                g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            return g
        elif return_type == 'pytorch_geometric_graph':
            edge_features = self.e_features_tensor[e_start: e_end].to(self.device)
            edge_indices = self.edge_indices[:, e_start: e_end].to(self.device)
            R_i = self.coordinates[start: start + n_atoms].to(self.device)
            z_i = self.features_tensor[start: start + n_atoms].to(self.device)
            return torch_geometric.data.Data(pos=R_i, z=z_i, edge_attr=edge_features, edge_index=edge_indices)
        elif return_type == 'pyg_multiple_conformers':
            edge_features = self.e_features_tensor[e_start: e_end]
            edge_indices = self.edge_indices[:, e_start: e_end]
            pos = self.conformations[start: start + n_atoms]
            features = self.features_tensor[start: start + n_atoms]
            pos_mask = self.pos_masks[idx]
            chiral_tag = self.chiral_tags[start: start + n_atoms]
            atoms_with_neighbors_start = self.meta_dict['atoms_with_neighbors_slices'][idx]
            atoms_with_neighbors_end = self.meta_dict['atoms_with_neighbors_slices'][
                                           idx + 1] + 1  # second +1 because the slices have a 0 appended to the beginning
            neighbor_slices = self.meta_dict['neighbors_slices'][atoms_with_neighbors_start:atoms_with_neighbors_end]
            neighbors_idx = self.meta_dict['neighbors_idx'][atoms_with_neighbors_start:atoms_with_neighbors_end]
            neighbors = {}
            for i, neighbor_slice in enumerate(neighbor_slices[:-1]):
                neighbors[neighbors_idx[i].item()] = self.meta_dict['neighbors_list'][neighbor_slice: neighbor_slices[i + 1]]
            return torch_geometric.data.Data(x=features, pos=pos, edge_attr=edge_features, edge_index=edge_indices,
                                             pos_mask=pos_mask, chiral_tag=chiral_tag, neighbors=neighbors)
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
        print('processing data from ({}) and saving it to ({})'.format(self.directory,
                                                                       os.path.join(self.directory, 'processed')))

        with open(os.path.join(self.directory, "summary_qm9.json"), "r") as f:
            summary = json.load(f)

        atom_slices = [0]
        edge_slices = [0]
        neighbors_slices = [0]
        neighbors_idx = []
        all_neighbors_list = []
        atoms_with_neighbors_slices = [0]
        total_eigvecs = []
        total_eigvals = []
        all_pos_masks = []
        all_atom_chiral_tags = []
        all_atom_features = []
        all_edge_features = []
        targets = {'ensembleenergy': [], 'ensembleentropy': [], 'ensemblefreeenergy': [], 'lowestenergy': [],
                   'poplowestpct': [], 'temperature': [], 'uniqueconfs': []}
        edge_indices = []  # edges of each molecule in coo format
        atomic_number_long = []
        n_atoms_list = []

        coordinates = []
        smiles_list = []
        total_atoms_with_neighbors = 0
        total_neighbors = 0
        total_atoms = 0
        total_edges = 0

        avg_degree = 0  # average degree in the dataset
        for smiles, sub_dic in tqdm(list(summary.items())):
            pickle_path = os.path.join(self.directory, sub_dic.get("pickle_path", ""))
            if os.path.isfile(pickle_path):
                pickle_file = open(pickle_path, 'rb')
                mol_dict = pickle.load(pickle_file)
                conformers = mol_dict['conformers']
                mol = conformers[0]['rd_mol']
                name = mol_dict["smiles"]
                n_atoms = mol.GetNumAtoms()
                # filter mols rdkit can't intrinsically handle
                mol_ = Chem.MolFromSmiles(name)
                if mol_:
                    canonical_smi = Chem.MolToSmiles(mol_)
                else:
                    continue

                # skip conformers with fragments
                if '.' in name or n_atoms < 4 or mol.GetNumBonds() < 4 or not mol.HasSubstructMatch(
                        dihedral_pattern) or 'ensembleenergy' not in mol_dict:
                    continue

                confs_coords = []
                pos_mask = torch.zeros(10, dtype=torch.int64)
                for i, conformer in enumerate(conformers[:10]):
                    conf_mol = conformer['rd_mol']

                    # skip conf_mols with atoms with more than 4 neighbors for now
                    n_neighbors = [len(a.GetNeighbors()) for a in conf_mol.GetAtoms()]
                    if np.max(n_neighbors) > 4:
                        continue

                    # filter for conformers that may have reacted
                    try:
                        conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(conf_mol))
                    except Exception as e:
                        continue

                    if conf_canonical_smi != canonical_smi:
                        continue

                    confs_coords.append(torch.tensor(conf_mol.GetConformer().GetPositions(), dtype=torch.float))
                    correct_mol = conf_mol
                    pos_mask[i] = 1

                if len(confs_coords) == 0:  # skip mol for which we have no valid conformers
                    continue
                if len(confs_coords) < 10:  # if there are less than 10 confs_coords we add the first one a few times
                    confs_coords.extend([confs_coords[0]] * (10 - len(confs_coords)))

                chiral_tag = []
                neighbors_list = []
                atom_features_list = []
                neighbor_dict = {}
                for i, atom in enumerate(correct_mol.GetAtoms()):
                    atom_features_list.append(atom_to_feature_vector(atom))
                    n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
                    if len(n_ids) > 1:
                        total_neighbors += len(n_ids)
                        neighbor_dict[i] = torch.tensor(n_ids)
                        neighbors_slices.append(total_neighbors)
                        neighbors_list.append(torch.tensor(n_ids, dtype=torch.long))
                        neighbors_idx.append(i)
                    chiral_tag.append(chirality[atom.GetChiralTag()])
                all_neighbors_list.append(torch.cat(neighbors_list, dim=0))
                total_atoms_with_neighbors += len(neighbors_list)
                atoms_with_neighbors_slices.append(
                    total_atoms_with_neighbors)  # we need this additionally because uusing atom_slices for slicing wont work since we only include the atoms with more than 1 neighbor

                all_atom_chiral_tags.append(torch.tensor(chiral_tag, dtype=torch.float))
                all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))
                adj = GetAdjacencyMatrix(correct_mol, useBO=False, force=True)
                max_freqs = 10
                adj = torch.tensor(adj).float()
                D = torch.diag(adj.sum(dim=0))
                L = D - adj
                N = adj.sum(dim=0) ** -0.5
                L_sym = torch.eye(n_atoms) - N * L * N
                try:
                    eig_vals, eig_vecs = torch.linalg.eigh(L_sym)
                except Exception as e:  # if we have disconnected components
                    deg = adj.sum(dim=0)
                    deg[deg == 0] = 1
                    N = deg ** -0.5
                    L_sym = torch.eye(n_atoms) - N * L * N
                    eig_vals, eig_vecs = torch.linalg.eigh(L_sym)
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
                edges_list = []
                edge_features_list = []
                for bond in correct_mol.GetBonds():
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

                targets['ensembleenergy'].append(mol_dict['ensembleenergy'])
                targets['ensembleentropy'].append(mol_dict['ensembleentropy'])
                targets['ensemblefreeenergy'].append(mol_dict['ensemblefreeenergy'])
                targets['lowestenergy'].append(mol_dict['lowestenergy'])
                targets['poplowestpct'].append(mol_dict['poplowestpct'])
                targets['temperature'].append(mol_dict['temperature'])
                targets['uniqueconfs'].append(mol_dict['uniqueconfs'])

                avg_degree += (len(edges_list) / 2) / n_atoms
                all_edge_features.append(edge_features)
                all_pos_masks.append(pos_mask)
                coordinates.append(torch.cat(confs_coords, dim=1))
                edge_indices.append(edge_index)
                total_edges += len(edges_list)
                total_atoms += n_atoms
                smiles_list.append(smiles)
                edge_slices.append(total_edges)
                atom_slices.append(total_atoms)
                n_atoms_list.append(n_atoms)

        for key, value in targets.items():
            targets[key] = torch.tensor(value)[:, None]
        data_dict = {'smiles': smiles_list,
                     'n_atoms': torch.tensor(n_atoms_list, dtype=torch.long),
                     'atom_slices': torch.tensor(atom_slices, dtype=torch.long),
                     'edge_slices': torch.tensor(edge_slices, dtype=torch.long),
                     'atom_features': torch.cat(all_atom_features, dim=0),
                     'edge_features': torch.cat(all_edge_features, dim=0),
                     'atomic_number_long': torch.tensor(atomic_number_long, dtype=torch.long),
                     'edge_indices': torch.cat(edge_indices, dim=1),
                     'coordinates': torch.cat(coordinates, dim=0).float(),
                     'targets': targets,
                     'neighbors_list': torch.cat(all_neighbors_list, dim=0),
                     'neighbors_idx': torch.tensor(neighbors_idx, dtype=torch.long),
                     # very long tensor with neighbors for each atom indexed by neighbors_slices
                     'neighbors_slices': torch.tensor(neighbors_slices, dtype=torch.long),
                     # n_atoms times num_neighbors where num_neighbors changes for each atom. Slice out of this with atoms_with_neighbors_slices
                     'atoms_with_neighbors_slices': torch.tensor(atoms_with_neighbors_slices, dtype=torch.long),
                     # n_atoms
                     'chiral_tags': torch.cat(all_atom_chiral_tags, dim=0),
                     'pos_masks': torch.stack(all_pos_masks, dim=0),
                     'avg_degree': avg_degree / len(n_atoms_list)
                     }


        data_dict.update(targets)
        if not os.path.exists(os.path.join(self.directory, 'processed')):
            os.mkdir(os.path.join(self.directory, 'processed'))
        torch.save(data_dict, os.path.join(self.directory, 'processed', self.processed_file))
