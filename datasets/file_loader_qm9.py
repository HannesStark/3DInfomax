import dgl
import networkx as nx
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType

import os.path as osp
import numpy as np
import glob
import pickle
import random

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset, Data, DataLoader
from tqdm import tqdm

from commons.geomol_utils import get_dihedral_pairs

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


class FileLoaderQM9(Dataset):
    def __init__(self, return_types=[], root='dataset/GEOM', transform=None, pre_transform=None, max_confs=10,
                 **kwargs):
        self.max_confs = max_confs
        super(FileLoaderQM9, self).__init__(root, transform, pre_transform)

        self.root = root
        self.return_types = return_types
        self.pickle_files = torch.load(self.processed_paths[0])
        self.dihedral_pairs = {}

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    @property
    def processed_file_names(self):
        return ['valid_files_qm9.pt']

    def process(self):
        valid_files = []
        for pickle_file in tqdm(sorted(glob.glob(osp.join(self.root, 'qm9', '*.pickle')))):
            mol_dic = self.open_pickle(pickle_file)
            data = self.featurize_mol(mol_dic)
            if data != None:
                valid_files.append(pickle_file)
        torch.save(valid_files, self.processed_paths[0])

    def len(self):
        return len(self.pickle_files)

    def get(self, idx):

        pickle_file = self.pickle_files[idx]
        mol_dic = self.open_pickle(pickle_file)
        data = self.featurize_mol(mol_dic)
        if idx in self.dihedral_pairs:
            data.edge_index_dihedral_pairs = self.dihedral_pairs[idx]
        else:
            data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, neighbors=None, data=data)

        if 'dgl_graph' in self.return_types:
            g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
            g.ndata['feat'] = data.x
            g.edata['feat'] = data.edge_attr
            return data, g
        else:
            return [data]

    def featurize_mol(self, mol_dic):
        confs = mol_dic['conformers']
        random.shuffle(confs)  # shuffle confs
        name = mol_dic["smiles"]

        # filter mols rdkit can't intrinsically handle
        mol_ = Chem.MolFromSmiles(name)
        if mol_:
            canonical_smi = Chem.MolToSmiles(mol_)
        else:
            return None

        # skip conformers with fragments
        if '.' in name:
            return None

        # skip conformers without dihedrals
        N = confs[0]['rd_mol'].GetNumAtoms()
        if N < 4:
            return None
        if confs[0]['rd_mol'].GetNumBonds() < 4:
            return None
        if not confs[0]['rd_mol'].HasSubstructMatch(dihedral_pattern):
            return None

        pos = torch.zeros([self.max_confs, N, 3])
        pos_mask = torch.zeros(self.max_confs, dtype=torch.int64)
        k = 0
        for conf in confs:
            mol = conf['rd_mol']

            # skip mols with atoms with more than 4 neighbors for now
            n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
            if np.max(n_neighbors) > 4:
                continue

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception as e:
                continue

            if conf_canonical_smi != canonical_smi:
                continue

            pos[k] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
            pos_mask[k] = 1
            k += 1
            correct_mol = mol
            if k == self.max_confs:
                break

        # return None if no non-reactive conformers were found
        if k == 0:
            return None

        atomic_number = []
        atom_features = []
        chiral_tag = []
        neighbor_dict = {}
        for i, atom in enumerate(correct_mol.GetAtoms()):
            n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(n_ids) > 1:
                neighbor_dict[i] = torch.tensor(n_ids)
            chiral_tag.append(chirality[atom.GetChiralTag()])
            atom_features.append(torch.tensor(atom_to_feature_vector(atom), dtype=torch.long))

        z = torch.tensor(atomic_number, dtype=torch.long)
        chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

        row, col, bond_features = [], [], []
        for bond in correct_mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_feature = torch.tensor(bond_to_feature_vector(bond), dtype=torch.long)
            bond_features.append(bond_feature)
            bond_features.append(bond_feature)

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.stack(bond_features, dim=0)

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

        x = torch.stack(atom_features, 0)

        data = Data(x=x, z=z, pos=[pos], edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict,
                    chiral_tag=chiral_tag, name=name, boltzmann_weight=conf['boltzmannweight'],
                    degeneracy=conf['degeneracy'], mol=correct_mol, pos_mask=pos_mask)
        return data
