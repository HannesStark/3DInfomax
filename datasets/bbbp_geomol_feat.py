import json
import os
import pickle

import torch
import dgl
from ogb.utils.features import bond_to_feature_vector, atom_to_feature_vector, get_atom_feature_dims, \
    get_bond_feature_dims
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType

import os.path as osp
import numpy as np
import glob
import pickle
import random
import pandas as pd

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset, Data, DataLoader, InMemoryDataset

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}


def one_k_encoding(value, choices):
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


class BBBPGeomol(InMemoryDataset):
    def __init__(self, split='train', root='dataset/bbbp', transform=None, pre_transform=None, device='cuda:0'):
        super(BBBPGeomol, self).__init__(root, transform, pre_transform)
        split_idx = ['train', 'val', 'test'].index(split)
        self.device = device
        self.data, self.slices = torch.load(self.processed_paths[split_idx])

    @property
    def raw_file_names(self):
        return ['BBBP.csv', 'bbbpscaffold123.pkl']

    @property
    def processed_file_names(self):
        return ['processed_train.pt', 'processed_val.pt', 'processed_test.pt']

    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)
        return data.to(self.device), torch.tensor([data.y]).to(self.device)

    def process(self):
        file = open(os.path.join(self.root, self.raw_file_names[1]), 'r')
        lines = file.readlines()
        split_idx = -2
        splits = [[], [], []]
        for index, line in enumerate(lines[:-1]):
            content = line.strip()
            if 'lp' in content:
                split_idx += 1
            else:
                splits[split_idx].append(int(content.strip('a').strip('I')))

        csv_file = pd.read_csv(os.path.join(self.root, self.raw_file_names[0]))
        for split_idx in [0, 1, 2]:
            data_list = []
            for i, smiles in enumerate(csv_file['smiles']):
                if i in splits[split_idx]:
                    try:
                        pyg_graph = featurize_mol_from_smiles(smiles)
                        pyg_graph.y = csv_file['p_np'][i]
                        data_list.append(pyg_graph)
                    except Exception as e:
                        print('rdkit failed for this smiles and it was excluded: ', smiles)
                        print('this was rdkits error message: ', e)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[split_idx])


bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
         'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
         'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
         'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}


def featurize_mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

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
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)
    chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

    row, col, edge_type, bond_features = [], [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bt = tuple(
            sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
        bond_features += 2 * [int(bond.IsInRing()),
                              int(bond.GetIsConjugated()),
                              int(bond.GetIsAromatic())]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    # bond_features = torch.tensor(bond_features, dtype=torch.float).view(len(bond_type), -1)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    # edge_attr = torch.cat([edge_attr[perm], bond_features], dim=-1)
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)

    data = Data(z=x, edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict, chiral_tag=chiral_tag,
                name=smiles, num_nodes=N)
    return data
