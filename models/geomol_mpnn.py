import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, TransformerEncoderLayer
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_sum

from models.base_layers import MLP


class GeomolMLP(nn.Module):
    """
    Creates a NN using nn.ModuleList to automatically adjust the number of layers.
    For each hidden layer, the number of inputs and outputs is constant.

    Inputs:
        in_dim (int):               number of features contained in the input layer.
        out_dim (int):              number of features input and output from each hidden layer,
                                    including the output layer.
        num_layers (int):           number of layers in the network
        activation (torch function): activation function to be used during the hidden layers
    """

    def __init__(self, in_dim, out_dim, num_layers, activation=torch.nn.ReLU(), layer_norm=False, batch_norm=False, batch_norm_momentum=0.1):
        super(GeomolMLP, self).__init__()
        self.layers = nn.ModuleList()

        h_dim = in_dim if out_dim < 10 else out_dim

        # create the input layer
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(nn.Linear(in_dim, h_dim))
            else:
                self.layers.append(nn.Linear(h_dim, h_dim))
            if layer_norm: self.layers.append(nn.LayerNorm(h_dim))
            if batch_norm: self.layers.append(nn.BatchNorm1d(h_dim, momentum=batch_norm_momentum))
            self.layers.append(activation)
        self.layers.append(nn.Linear(h_dim, out_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class GeomolMetaLayer(torch.nn.Module):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.
    """

    def __init__(self, edge_model=None, node_model=None):
        super(GeomolMetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model

        self.edge_eps = nn.Parameter(torch.Tensor([0]))
        self.node_eps = nn.Parameter(torch.Tensor([0]))

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """"""
        if self.edge_model is not None:
            edge_attr = (1 + self.edge_eps) * edge_attr + self.edge_model(x, edge_attr, edge_index)
        if self.node_model is not None:
            x = (1 + self.node_eps) * x + self.node_model(x, edge_index, edge_attr, batch)

        return x, edge_attr


class EdgeModel(nn.Module):
    def __init__(self, hidden_dim, n_layers, batch_norm_momentum=0.1):
        super(EdgeModel, self).__init__()
        self.edge = Lin(hidden_dim, hidden_dim)
        self.node_in = Lin(hidden_dim, hidden_dim, bias=False)
        self.node_out = Lin(hidden_dim, hidden_dim, bias=False)
        self.mlp = GeomolMLP(hidden_dim, hidden_dim, n_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, x, edge_attr, edge_index):
        # source, target: [2, E], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs (we don't have any of these yet)
        # batch: [E] with max entry B - 1.

        f_ij = self.edge(edge_attr)
        f_i = self.node_in(x)
        f_j = self.node_out(x)
        row, col = edge_index

        out = F.relu(f_ij + f_i[row] + f_j[col])
        return self.mlp(out)


class GeomolNodeModel(nn.Module):
    def __init__(self, hidden_dim, n_layers, batch_norm_momentum=0.1):
        super(GeomolNodeModel, self).__init__()
        self.node_mlp_1 = GeomolMLP(hidden_dim, hidden_dim, n_layers, batch_norm_momentum=batch_norm_momentum)
        self.node_mlp_2 = GeomolMLP(hidden_dim, hidden_dim, n_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [N, h], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u] (N/A)
        # batch: [N] with max entry B - 1.
        # source, target = edge_index
        _, col = edge_index
        out = self.node_mlp_1(edge_attr)
        out = scatter_sum(out, col, dim=0, dim_size=x.size(0))
        return self.node_mlp_2(out)


class GeomolGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=300, depth=3, n_layers=2):
        super(GeomolGNN, self).__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.node_init = GeomolMLP(node_dim, hidden_dim, n_layers)
        self.edge_init = GeomolMLP(edge_dim, hidden_dim, n_layers)
        self.update = GeomolMetaLayer(EdgeModel(hidden_dim, n_layers), GeomolNodeModel(hidden_dim, n_layers))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_init(x)
        edge_attr = self.edge_init(edge_attr)
        for _ in range(self.depth):
            x, edge_attr = self.update(x, edge_index, edge_attr)
        return x, edge_attr


class GeomolGNNWrapper(nn.Module):
    def __init__(self, hidden_dim, node_dim, edge_dim, readout_layers=2, readout_batchnorm=True, **kwargs):
        super(GeomolGNNWrapper, self).__init__()

        self.random_vec_dim = 10
        self.random_vec_std = 1.0

        self.gnn = GeomolGNN(hidden_dim=hidden_dim, node_dim=node_dim +10, edge_dim=edge_dim + 10, **kwargs)
        self.output = MLP(in_dim=hidden_dim, hidden_size=hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=1,
                          layers=readout_layers, batch_norm_momentum=0.1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.z, data.edge_index, data.edge_attr, data.batch

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        # rand_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        rand_x = rand_dist.sample([x.size(0), self.random_vec_dim]).squeeze(-1).to(
            x.device)  # added squeeze
        rand_edge = rand_dist.sample([edge_attr.size(0), self.random_vec_dim]).squeeze(-1).to(
            x.device)  # added squeeze
        x = torch.cat([x, rand_x], dim=-1)
        edge_attr = torch.cat([edge_attr, rand_edge], dim=-1)

        x, edge_attr = self.gnn(x, edge_index, edge_attr)
        pooled = global_mean_pool(x, batch)
        return self.output(pooled)

