import copy
from typing import List, Tuple

import dgl
import torch
from torch.nn.utils.rnn import pad_sequence

from commons.utils import get_adj_matrix


def graph_collate(batch: List[Tuple]):
    graphs, targets = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return [batched_graph], torch.stack(targets).float()



def s_norm_graph_collate(batch: List[Tuple]):
    graphs, targets = map(list, zip(*batch))
    tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
    snorm_n = torch.cat(tab_snorm_n).sqrt()
    batched_graph = dgl.batch(graphs)
    return [batched_graph], snorm_n, torch.stack(targets).float()


def pairwise_distance_collate(batch: List[Tuple]):
    mol_graphs, pairwise_indices, distances = map(list, zip(*batch))
    batched_mol_graph = dgl.batch(mol_graphs)

    return [batched_mol_graph], [torch.cat(pairwise_indices, dim=-1)], torch.cat(distances)


def s_norm_contrastive_collate(batch: List[Tuple]):
    # optionally take targets
    graphs, graphs3d = map(list, zip(*batch))
    tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
    snorm_n = torch.cat(tab_snorm_n).sqrt()
    batched_graph = dgl.batch(graphs)
    batched_graph3d = dgl.batch(graphs3d)

    return [batched_graph], [batched_graph3d], snorm_n

def contrastive_collate(batch: List[Tuple]):
    # optionally take targets
    graphs, graphs3d, *targets = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    batched_graph3d = dgl.batch(graphs3d)

    if targets:
        return [batched_graph], [batched_graph3d], torch.stack(*targets).float()
    else:
        return [batched_graph], [batched_graph3d]


class NoisedDistancesCollate(object):
    def __init__(self, std, num_noised):
        self.std = std
        self.num_noised = num_noised

    def __call__(self, batch: List[Tuple]):

        graphs, graphs3d, *targets = map(list, zip(*batch))
        batched_graph = dgl.batch(graphs)
        batched_graph3d = dgl.batch(graphs3d)
        graphs3d_noised = [batched_graph3d]
        for i in range(self.num_noised):
            copy_graph = copy.deepcopy(batched_graph3d)
            copy_graph.edata['w'] += torch.randn_like(copy_graph.edata['w']) * self.std
            graphs3d_noised.append(copy_graph)

        batched_graph3d = dgl.batch(graphs3d_noised)

        if targets:
            return [batched_graph], [batched_graph3d], torch.stack(*targets).float()
        else:
            return [batched_graph], [batched_graph3d]


class ConformerCollate(object):
    def __init__(self, num_conformers):
        self.num_conformers = num_conformers

    def __call__(self, batch: List[Tuple]):
        graphs, graphs3d, conformers, *targets = map(list, zip(*batch))
        conformers = torch.cat(conformers, dim=0)
        batched_graph3d = dgl.batch(graphs3d)
        conformer_graphs = [batched_graph3d]
        for i in range(1, self.num_conformers):
            conformer_graph = copy.deepcopy(batched_graph3d)
            conformer_graph.ndata['x'] = conformers[:, i * 3:(i + 1) * 3]
            conformer_graphs.append(conformer_graph)
        batched_conformers = dgl.batch(conformer_graphs)
        batched_graph = dgl.batch(graphs)

        if targets:
            return [batched_graph], [batched_conformers], torch.stack(*targets).float()
        else:
            return [batched_graph], [batched_conformers]



class NoisedCoordinatesCollate(object):
    def __init__(self, std, num_noised):
        self.std = std
        self.num_noised = num_noised

    def __call__(self, batch: List[Tuple]):

        graphs, graphs3d, *targets = map(list, zip(*batch))
        batched_graph = dgl.batch(graphs)
        batched_graph3d = dgl.batch(graphs3d)
        graphs3d_noised = [batched_graph3d]
        previous_distances = batched_graph3d.edata['w']
        edges = batched_graph3d.all_edges()
        for i in range(self.num_noised):
            copy_graph = copy.deepcopy(batched_graph3d)
            copy_graph.ndata['x'] += torch.randn_like(copy_graph.ndata['x']) * self.std
            distances = torch.norm(copy_graph.ndata['x'][edges[0]] - copy_graph.ndata['x'][edges[1]], p=2, dim=-1)
            copy_graph.edata['w'] = distances[:, None]
            graphs3d_noised.append(copy_graph)

        batched_graph3d = dgl.batch(graphs3d_noised)

        if targets:
            return [batched_graph], [batched_graph3d], torch.stack(*targets).float()
        else:
            return [batched_graph], [batched_graph3d]


class NodeDrop3dCollate(object):
    def __init__(self, num_drop):
        self.num_drop = num_drop

    def __call__(self, batch: List[Tuple]):
        graphs, graphs3d = map(list, zip(*batch))
        device = graphs3d[0].device
        for graph3d in graphs3d:
            remove_number = torch.randint(low=0, high=self.num_drop, size=(1,))
            if remove_number > 0:
                remove_indices = torch.randint(low=0, high=graph3d.number_of_nodes(), size=(remove_number.data,),
                                               device=device)
                graph3d.remove_nodes(remove_indices)
        batched_graph = dgl.batch(graphs)
        batched_graph3d = dgl.batch(graphs3d)

        return [batched_graph], [batched_graph3d]


class NodeDrop2dCollate(object):
    def __init__(self, num_drop):
        self.num_drop = num_drop

    def __call__(self, batch: List[Tuple]):
        graphs, graphs3d = map(list, zip(*batch))
        device = graphs3d[0].device
        for graph in graphs:
            remove_number = torch.randint(low=0, high=self.num_drop, size=(1,))
            if remove_number > 0:
                remove_indices = torch.randint(low=0, high=graph.number_of_nodes(), size=(remove_number.data,),
                                               device=device)
                graph.remove_nodes(remove_indices)
        batched_graph = dgl.batch(graphs)
        batched_graph3d = dgl.batch(graphs3d)

        return [batched_graph], [batched_graph3d]


def padded_collate(batch):
    """
    Takes list of tuples with molecule features of variable sizes (different n_atoms) and pads them with zeros for processing as a sequence
    Args:
        batch: list of tuples with embeddings and the corresponding label
    """

    features = pad_sequence([item[0] for item in batch], batch_first=True)
    targets = torch.stack([item[1] for item in batch])

    # create mask corresponding to the zero padding used for the shorter sequences in the batch.
    # All values corresponding to padding are True and the rest is False.
    n_atoms = torch.tensor([len(item[0]) for item in batch])
    mask = torch.arange(features.shape[1])[None, :] >= n_atoms[:, None]  # [batch_size, n_atoms]
    return [features, mask], targets

def egnn_padded_collate3d(batch):
    graphs, features, coordinates = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)

    # pad features with -1 because that is used as padding index in the atom embedder in egnn
    features = pad_sequence(features, batch_first=True, padding_value=-1)
    coordinates = pad_sequence(coordinates, batch_first=True, padding_value=0)
    atom_mask = features.sum(-1) > -1

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=features.device).unsqueeze(0)
    edge_mask *= diag_mask

    edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1).float()

    edges = get_adj_matrix(n_nodes, batch_size, device=features.device)

    features = features.view(batch_size * n_nodes, -1)
    coordinates = coordinates.view(batch_size * n_nodes, -1)
    atom_mask = atom_mask.view(batch_size * n_nodes, -1).float()
    return [batched_graph], [features, coordinates, edges, None, atom_mask, edge_mask, n_nodes]

def egnn_padded_collate(batch):
    features, coordinates, targets = map(list, zip(*batch))

    # pad features with -1 because that is used as padding index in the atom embedder in egnn
    features = pad_sequence(features, batch_first=True, padding_value=-1)
    coordinates = pad_sequence(coordinates, batch_first=True, padding_value=0)
    atom_mask = features.sum(-1) > -1

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=features.device).unsqueeze(0)
    edge_mask *= diag_mask

    edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1).float()

    edges = get_adj_matrix(n_nodes, batch_size, device=features.device)

    features = features.view(batch_size * n_nodes, -1)
    coordinates = coordinates.view(batch_size * n_nodes, -1)
    atom_mask = atom_mask.view(batch_size * n_nodes, -1).float()
    return [features, coordinates, edges, None, atom_mask, edge_mask, n_nodes], torch.stack(targets)

def padded_collate_positional_encoding(batch):
    """
    Takes list of tuples with molecule features of variable sizes (different n_atoms) and pads them with zeros for processing as a sequence
    Args:
        batch: list of tuples with embeddings and the corresponding label
    """

    features = pad_sequence([item[0] for item in batch], batch_first=True)
    pos_enc = pad_sequence([item[1] for item in batch], batch_first=True)
    targets = torch.stack([item[2] for item in batch])

    # create mask corresponding to the zero padding used for the shorter sequences in the batch.
    # All values corresponding to padding are True and the rest is False.
    n_atoms = torch.tensor([len(item[0]) for item in batch])
    mask = torch.arange(features.shape[1])[None, :] >= n_atoms[:, None]  # [batch_size, n_atoms]
    return [features, pos_enc, mask], targets


def molhiv_padded_collate(batch: List[Tuple]):
    graphs, targets = map(list, zip(*batch))
    features = pad_sequence([graph.ndata['feat'] for graph in graphs], batch_first=True)

    n_atoms = torch.tensor([graph.number_of_nodes() for graph in graphs])
    mask = torch.arange(features.shape[1])[None, :] >= n_atoms[:, None]  # [batch_size, n_atoms]
    return [features, None, mask], torch.stack(targets)

def padded_distances_collate(batch):
    """
    Takes list of tuples with molecule features of variable sizes (different n_atoms) and pads them with zeros for processing as a sequence
    Args:
        batch: list of tuples with embeddings and the corresponding label
    """
    graphs, distances = map(list, zip(*batch))
    padded = pad_sequence(distances, batch_first=True)

    # create mask corresponding to the zero padding used for the shorter sequences in the batch.
    # All values corresponding to padding are True and the rest is False.
    n_dist = torch.tensor([len(dist) for dist in distances])
    mask = torch.arange(padded.shape[1])[None, :] >= n_dist[:, None]  # [batch_size, n_atoms]
    batched_graph = dgl.batch(graphs)
    return batched_graph, padded, mask
