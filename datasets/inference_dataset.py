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


class InferenceDataset(Dataset):

    def __init__(self, smiles_txt_path, device='cuda:0', transform=None, **kwargs):
        with open(smiles_txt_path) as file:
            lines = file.readlines()
            smiles_list = [line.rstrip() for line in lines]
        atom_slices = [0]
        edge_slices = [0]
        all_atom_features = []
        all_edge_features = []
        edge_indices = []  # edges of each molecule in coo format
        total_atoms = 0
        total_edges = 0
        n_atoms_list = []
        for mol_idx, smiles in tqdm(enumerate(smiles_list)):
            # get the molecule using the smiles representation from the csv file
            mol = Chem.MolFromSmiles(smiles)
            # add hydrogen bonds to molecule because they are not in the smiles representation
            mol = Chem.AddHs(mol)
            n_atoms = mol.GetNumAtoms()

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

            edge_indices.append(edge_index)
            all_edge_features.append(edge_features)
            total_edges += len(edges_list)
            total_atoms += n_atoms
            edge_slices.append(total_edges)
            atom_slices.append(total_atoms)
            n_atoms_list.append(n_atoms)

        self.n_atoms = torch.tensor(n_atoms_list)
        self.atom_slices = torch.tensor(atom_slices, dtype=torch.long)
        self.edge_slices = torch.tensor(edge_slices, dtype=torch.long)
        self.edge_indices = torch.cat(edge_indices, dim=1)
        self.all_atom_features = torch.cat(all_atom_features, dim=0)
        self.all_edge_features = torch.cat(all_edge_features, dim=0)

    def __len__(self):
        return len(self.atom_slices) - 1

    def __getitem__(self, idx):

        e_start = self.edge_slices[idx]
        e_end = self.edge_slices[idx + 1]
        start = self.atom_slices[idx]
        n_atoms = self.n_atoms[idx]
        edge_indices = self.edge_indices[:, e_start: e_end]
        g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms)
        g.ndata['feat'] = self.all_atom_features[start: start + n_atoms]
        g.edata['feat'] = self.all_edge_features[e_start: e_end]
        return g
