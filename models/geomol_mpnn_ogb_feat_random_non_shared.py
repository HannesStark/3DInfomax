import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, TransformerEncoderLayer
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_sum

from commons.mol_encoder import BondEncoder, AtomEncoder
from models.base_layers import MLP
from models.geomol_mpnn import GeomolMLP, GeomolMetaLayer, EdgeModel, GeomolNodeModel


class GeomolGNNOGBFeatRandomNonShared(nn.Module):
    def __init__(self, random_vec_dim, n_model_confs=None, hidden_dim=300, depth=3, n_layers=2, batch_norm_momentum=0.1,
                 pretrain_mode=False, **kwargs):
        super(GeomolGNNOGBFeatRandomNonShared, self).__init__()

        self.n_model_confs = n_model_confs
        self.depth = depth
        self.pretrain_mode = pretrain_mode
        self.hidden_dim = hidden_dim
        self.bond_encoder = BondEncoder(hidden_dim)
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.node_init = GeomolMLP(hidden_dim + random_vec_dim, hidden_dim, num_layers=2,
                                   batch_norm_momentum=batch_norm_momentum)
        self.edge_init = GeomolMLP(hidden_dim + random_vec_dim, hidden_dim, num_layers=2,
                                   batch_norm_momentum=batch_norm_momentum)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(GeomolMetaLayer(EdgeModel(hidden_dim, n_layers, batch_norm_momentum=batch_norm_momentum),
                                      GeomolNodeModel(hidden_dim, n_layers, batch_norm_momentum=batch_norm_momentum)))

    def forward(self, x, edge_index, edge_attr, rand_x, rand_edge, **kwargs):
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        if self.pretrain_mode:
            x = x.unsqueeze(1).repeat(1, self.n_model_confs, 1)
            edge_attr = edge_attr.unsqueeze(1).repeat(1, self.n_model_confs, 1)
        x = torch.cat([x, rand_x], dim=-1)
        edge_attr = torch.cat([edge_attr, rand_edge], dim=-1)

        x = self.node_init(x)
        edge_attr = self.edge_init(edge_attr)
        for i in range(self.depth):
            x, edge_attr = self.layers[i](x, edge_index, edge_attr)
        return x, edge_attr


class GeomolGNNWrapperOGBFeatRandomNonShared(nn.Module):
    def __init__(self, hidden_dim, target_dim, gnn_params, readout_hidden_dim=None, readout_layers=2,
                 readout_batchnorm=True, random_vec_dim=10, random_vec_std=1.0,
                 **kwargs):
        super(GeomolGNNWrapperOGBFeatRandomNonShared, self).__init__()

        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.node_gnn = GeomolGNNOGBFeatRandomNonShared(random_vec_dim=random_vec_dim, **gnn_params)
        self.output = MLP(in_dim=hidden_dim, hidden_size=readout_hidden_dim, mid_batch_norm=readout_batchnorm,
                          out_dim=target_dim, layers=readout_layers, batch_norm_momentum=0.1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.z, data.edge_index, data.edge_attr, data.batch

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        # rand_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        rand_x = rand_dist.sample([x.size(0), self.random_vec_dim]).squeeze(-1).to(x.device)
        rand_edge = rand_dist.sample([edge_attr.size(0), self.random_vec_dim]).squeeze(-1).to(x.device)

        x, edge_attr = self.node_gnn(x, edge_index, edge_attr, rand_x, rand_edge)
        pooled = global_mean_pool(x, batch)
        return self.output(pooled)
