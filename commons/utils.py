import os
import random
from argparse import Namespace
from collections import MutableMapping
from typing import Dict, Any
import matplotlib.pyplot as plt
import sklearn

import torch
import numpy as np
import dgl
from torch.utils.tensorboard import SummaryWriter


def seed_all(seed):
    if not seed:
        seed = 0

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_random_indices(length, seed=123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def flatten_dict(params: Dict[Any, Any], delimiter: str = '/') -> Dict[str, Any]:
    """
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.
    Examples:
        flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    for d in _dict_generator(value, prefixes + [key]):
                        yield d
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    dictionary = {delimiter.join(keys): val for *keys, val in _dict_generator(params)}
    for k in dictionary.keys():
        # convert relevant np scalars to python types first (instead of str)
        if isinstance(dictionary[k], (np.bool_, np.integer, np.floating)):
            dictionary[k] = dictionary[k].item()
        elif type(dictionary[k]) not in [bool, int, float, str, torch.Tensor]:
            dictionary[k] = str(dictionary[k])
    return dictionary


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze()


def tensorboard_singular_value_plot(predictions, targets, writer: SummaryWriter, step, data_split):
    u, s, v = torch.pca_lowrank(predictions.detach().cpu(), q=min(predictions.shape))
    fig, ax = plt.subplots()
    s = 100 * s / s.sum()
    ax.plot(s.numpy())
    writer.add_figure(f'singular_values/{data_split}', figure=fig, global_step=step)
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(s.numpy()))
    writer.add_figure(f'singular_values_cumsum/{data_split}', figure=fig, global_step=step)


def tensorboard_gradient_magnitude(optimizer: torch.optim.Optimizer, writer: SummaryWriter, step, param_groups=[0]):
    for i, param_group in enumerate(optimizer.param_groups):
        if i in param_groups:
            all_params = []
            for params in param_group['params']:
                if params.grad != None:
                    all_params.append(params.grad.view(-1))
            writer.add_scalar(f'gradient_magnitude_param_group_{i}', torch.cat(all_params).abs().mean(),
                              global_step=step)


TENSORBOARD_FUNCTIONS = {
    'singular_values': tensorboard_singular_value_plot
}

def move_to_device(element, device):
    '''
    takes arbitrarily nested list and moves everything in it to device if it is a dgl graph or a torch tensor
    :param element: arbitrarily nested list
    :param device:
    :return:
    '''
    if isinstance(element, list):
        return [move_to_device(x, device) for x in element]
    else:
        return element.to(device) if isinstance(element,(torch.Tensor, dgl.DGLGraph)) else element

