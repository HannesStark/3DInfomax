from typing import List, Tuple

import dgl
import torch
from torch.nn.utils.rnn import pad_sequence


def graph_collate(batch: List[Tuple]):
    graphs, targets = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(targets)


def pairwise_distance_collate(batch: List[Tuple]):
    mol_graphs, pairwise_indices, distances = map(list, zip(*batch))
    batched_mol_graph = dgl.batch(mol_graphs)

    return batched_mol_graph, torch.cat(pairwise_indices, dim=-1), torch.cat(distances)


def contrastive_collate(batch: List[Tuple]):
    graphs, graphs3d = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    batched_graph3d = dgl.batch(graphs3d)

    return batched_graph, batched_graph3d

def random_3d_node_drop_collate(batch: List[Tuple]):
    graphs, graphs3d = map(list, zip(*batch))
    device = graphs3d[0].device
    for graph3d in graphs3d:
        remove_number = torch.randint(low=0, high=5, size=(1,))
        if remove_number > 0:
            remove_indices = torch.randint(low=0, high=graph3d.number_of_nodes(), size=(remove_number.data,), device=device)
            graph3d.remove_nodes(remove_indices)
    batched_graph = dgl.batch(graphs)
    batched_graph3d = dgl.batch(graphs3d)

    return batched_graph, batched_graph3d


def padded_collate(batch):
    """
    Takes list of tuples with molecule features of variable sizes (different n_atoms) and pads them with zeros for processing as a sequence
    Args:
        batch: list of tuples with embeddings and the corresponding label
    """

    features = pad_sequence([item[0] for item in batch], batch_first=True)
    target = torch.stack([item[1] for item in batch])

    # create mask corresponding to the zero padding used for the shorter sequences in the batch.
    # All values corresponding to padding are True and the rest is False.
    n_atoms = torch.tensor([len(item[2]) for item in batch])
    mask = torch.arange(features.shape[1])[None, :] >= n_atoms[:, None]  # [batch_size, n_atoms]
    return features, mask, target
