from collections import defaultdict
from typing import List, Optional

import torch
from torch.distributions import Categorical
from torch.utils.data import Sampler, RandomSampler, Subset
from tqdm import tqdm

from datasets.qm9_dataset import QM9Dataset


class ConstantNumberAtomsCategorical(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    """

    def __init__(self, data_source: QM9Dataset, batch_size: int, indices: List, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, drop_last=False) -> None:
        super(Sampler, self).__init__()
        n_atoms = data_source.meta_dict['n_atoms'][indices]
        self.data_source = Subset(data_source, indices)
        self.clusters = defaultdict(list)
        for i, n_atom in tqdm(enumerate(n_atoms)):
            self.clusters[n_atom.item()].append(i)
        print(self.clusters)
        self.categorical = Categorical(
            probs=torch.tensor([len(v) for v in self.clusters.values()]) / len(self.data_source))
        self.sampler = RandomSampler(data_source=self.data_source, replacement=replacement, num_samples=num_samples,
                                     generator=generator)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        n_atoms = list(self.clusters.keys())[self.categorical.sample()]
        cluster = self.clusters[n_atoms]
        cluster_indices = list(torch.randperm(len(cluster)))

        for idx in self.sampler:
            if len(batch) < self.batch_size // 2 and cluster_indices != []:
                batch.append(cluster[cluster_indices.pop(0)])
            else:
                batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                n_atoms = list(self.clusters.keys())[self.categorical.sample()]
                cluster = self.clusters[n_atoms]
                cluster_indices = list(torch.randperm(len(cluster)))
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        return (len(self.data_source) + self.batch_size - 1) // self.batch_size  # type: ignore


class ConstantNumberAtomsChunks(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    """

    def __init__(self, data_source: QM9Dataset, batch_size: int, indices: List, drop_last=False,
                 number_chunks=50) -> None:
        super(Sampler, self).__init__()
        self.number_chunks = number_chunks
        n_atoms = data_source.meta_dict['n_atoms'][indices]
        self.data_source = Subset(data_source, indices)
        n_atoms_sorted, self.indices_sorted_by_n_atoms = torch.sort(n_atoms)

        # get indices between which the molecules with the sambe number of atoms lie
        self.n_atoms_separators = [0]
        max = n_atoms_sorted[0].item()
        for i, n_atoms in enumerate(n_atoms_sorted):
            if max < n_atoms:
                self.n_atoms_separators.append(i)
                max = n_atoms
        self.n_atoms_separators.append(len(n_atoms_sorted))

        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        clusters = self.get_clusters()
        next_cluster = 1
        for _ in range(len(self.data_source)):
            if len(batch) < self.batch_size // 2 and clusters[0] != []:
                batch.append(clusters[0].pop(0))
            else:
                if clusters[next_cluster] != []:
                    batch.append(clusters[next_cluster].pop(0))
                else:
                    next_cluster += 1
                    batch.append(clusters[next_cluster].pop(0))

            if len(batch) == self.batch_size:
                yield batch
                clusters = self.get_clusters()
                next_cluster = 1
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def get_clusters(self):
        shuffled_among_same_n_atoms = []
        for i, _ in enumerate(self.n_atoms_separators[:-1]):
            same_n_atoms = self.indices_sorted_by_n_atoms[self.n_atoms_separators[i]:self.n_atoms_separators[i + 1]]
            perm = torch.randperm(len(same_n_atoms))
            shuffled_among_same_n_atoms.append(same_n_atoms[perm])
        shuffled_among_same_n_atoms = torch.cat(shuffled_among_same_n_atoms)
        clusters = torch.chunk(shuffled_among_same_n_atoms, self.number_chunks)
        cluster_choices = torch.stack(clusters)[torch.randperm(len(clusters)).tolist()]

        return cluster_choices.tolist()

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        return (len(self.data_source) + self.batch_size - 1) // self.batch_size  # type: ignore
