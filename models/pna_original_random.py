import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from commons.mol_encoder import AtomEncoder, BondEncoder

EPS = 1e-5
import numpy as np
from models.base_layers import MLP, MLPReadout

"""
    code from:
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output

def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (np.log(D + 1) / avg_d)


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    return h * (avg_d / np.log(D + 1))


SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation}


def aggregate_mean(h):
    return torch.mean(h, dim=1)


def aggregate_max(h):
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    return torch.min(h, dim=1)[0]


def aggregate_std(h):
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


class GRU(nn.Module):
    """
        Wrapper class for the GRU used by the GNN framework, nn.GRU is used for the Gated Recurrent Unit itself
    """

    def __init__(self, input_size, hidden_size, device):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size).to(device)

    def forward(self, x, y):
        """
        :param x:   shape: (B, N, Din) where Din <= input_size (difference is padded)
        :param y:   shape: (B, N, Dh) where Dh <= hidden_size (difference is padded)
        :return:    shape: (B, N, Dh)
        """
        assert (x.shape[-1] <= self.input_size and y.shape[-1] <= self.hidden_size)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = self.gru(x, y)[1]
        x = x.squeeze()
        return x


def aggregate_moment(h, n=3):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n))
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1. / n)
    return rooted_h_n


def aggregate_moment_3(h):
    return aggregate_moment(h, n=3)


def aggregate_moment_4(h):
    return aggregate_moment(h, n=4)


def aggregate_moment_5(h):
    return aggregate_moment(h, n=5)


def aggregate_sum(h):
    return torch.sum(h, dim=1)


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var, 'moment3': aggregate_moment_3, 'moment4': aggregate_moment_4,
               'moment5': aggregate_moment_5}


class PNAOriginal(nn.Module):
    def __init__(self, hidden_dim, last_layer_dim, target_dim, in_feat_dropout, dropout, last_batch_norm,
                 mid_batch_norm, propagation_depth, readout_aggregators, readout_hidden_dim, readout_layers,
                 aggregators, scalers, avg_d, residual, posttrans_layers, pretrans_layers, device, edge_hidden_dim,
                 graph_norm, use_3d=False, gru_enable=False, divide_input_last=True, divide_input_first=True,
                 edge_feat=True, towers=1, **kwargs):
        super().__init__()

        self.node_gnn = PNAGNNOriginal(hidden_dim=hidden_dim, last_layer_dim=last_layer_dim, last_batch_norm=last_batch_norm,
                                  mid_batch_norm=mid_batch_norm, in_feat_dropout=in_feat_dropout, dropout=dropout,
                                  aggregators=aggregators, scalers=scalers, residual=residual, avg_d=avg_d,
                                  propagation_depth=propagation_depth, posttrans_layers=posttrans_layers, device=device,
                                  pretrans_layers=pretrans_layers, gru_enable=gru_enable, use_3d=use_3d,
                                  edge_hidden_dim=edge_hidden_dim, divide_input_first=divide_input_first,
                                  divide_input_last=divide_input_last, edge_feat=edge_feat, graph_norm=graph_norm,
                                  towers=towers)

        self.readout_aggregators = readout_aggregators
        self.output = MLPReadout(last_layer_dim * len(self.readout_aggregators), target_dim)

    def forward(self, g, snorm_n):
        h = g.ndata['feat']
        e = g.edata['feat']
        g, h = self.node_gnn(g, h, e, snorm_n)

        readouts_to_cat = [dgl.readout_nodes(g, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class PNAGNNOriginal(nn.Module):
    def __init__(self, hidden_dim, last_layer_dim, in_feat_dropout, dropout, propagation_depth, graph_norm,
                 mid_batch_norm, last_batch_norm, residual, aggregators, scalers, avg_d, use_3d,
                 towers, divide_input_first, divide_input_last, edge_feat, edge_hidden_dim, pretrans_layers,
                 posttrans_layers, gru_enable, device):
        super().__init__()
        self.gru_enable = gru_enable
        self.edge_feat = edge_feat

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = AtomEncoder(hidden_dim)
        if self.edge_feat:
            self.embedding_e = BondEncoder(edge_hidden_dim)

        self.layers = nn.ModuleList([PNALayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                                              graph_norm=graph_norm, mid_batch_norm=mid_batch_norm,
                                              last_batch_norm=last_batch_norm, use_3d=use_3d,
                                              residual=residual, aggregators=aggregators, scalers=scalers,
                                              avg_d=avg_d, towers=towers, edge_features=edge_feat,
                                              edge_hidden_dim=edge_hidden_dim, divide_input=divide_input_first,
                                              pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers) for _
                                     in range(propagation_depth - 1)])
        self.layers.append(PNALayer(in_dim=hidden_dim, out_dim=last_layer_dim, dropout=dropout,
                                    graph_norm=graph_norm, mid_batch_norm=mid_batch_norm,
                                    last_batch_norm=last_batch_norm, use_3d=use_3d,
                                    residual=residual, aggregators=aggregators, scalers=scalers,
                                    avg_d=avg_d, towers=towers, divide_input=divide_input_last,
                                    edge_features=edge_feat, edge_hidden_dim=edge_hidden_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers))

        if self.gru_enable:
            self.gru = GRU(hidden_dim, hidden_dim, device)

        self.MLP_layer = MLPReadout(hidden_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, h, e, snorm_n):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.edge_feat:
            e = self.embedding_e(e)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t

        g.ndata['feat'] = h
        return g, h


class PNATower(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, mid_batch_norm, last_batch_norm, aggregators, scalers,
                 avg_d, use_3d, pretrans_layers, posttrans_layers, edge_features, edge_hidden_dim):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.graph_norm = graph_norm
        self.edge_features = edge_features
        self.use_3d = use_3d

        self.aggregators = aggregators
        self.scalers = scalers
        self.pretrans = MLP(in_dim=2 * in_dim + (edge_hidden_dim if edge_features else 0) + (1 if use_3d else 0),
                            hidden_size=in_dim, out_dim=in_dim, layers=pretrans_layers, mid_activation='relu',
                            last_activation='none')
        self.posttrans = MLP(in_dim=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim,
                             mid_batch_norm=mid_batch_norm, last_batch_norm=last_batch_norm,
                             out_dim=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1)
        else:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)
        if self.use_3d:
            distances = torch.norm((edges.src['x'] - edges.dst['x']), dim=-1)[:, None]
            z2 = torch.cat([z2, distances], dim=1)
        return {'e': self.pretrans(z2)}

    def message_func(self, edges):
        return {'e': edges.data['e']}

    def reduce_func(self, nodes):
        h = nodes.mailbox['e']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'feat': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['feat'])

    def forward(self, g, h, e, snorm_n):
        g.ndata['feat'] = h
        if self.edge_features:  # add the edges information only if edge_features = True
            g.edata['feat'] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['feat']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization
        if self.graph_norm:
            h = h * snorm_n
        h = self.dropout(h)
        return h


class PNALayer(nn.Module):

    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, graph_norm, mid_batch_norm, use_3d,
                 last_batch_norm, towers=1,
                 pretrans_layers=1, posttrans_layers=1, divide_input=True, residual=False, edge_features=False,
                 edge_hidden_dim=0):

        super().__init__()
        assert ((
                    not divide_input) or in_dim % towers == 0), "if divide_input is set the number of towers has to divide in_dim"
        assert (out_dim % towers == 0), "the number of towers has to divide the last_layer_dim"
        assert avg_d is not None

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        scalers = [SCALERS[scale] for scale in scalers]

        self.divide_input = divide_input
        self.input_tower = in_dim // towers if divide_input else in_dim
        self.output_tower = out_dim // towers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        # convolution
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(PNATower(in_dim=self.input_tower, out_dim=self.output_tower, aggregators=aggregators,
                                        scalers=scalers, avg_d=avg_d, pretrans_layers=pretrans_layers,
                                        posttrans_layers=posttrans_layers, mid_batch_norm=mid_batch_norm,
                                        last_batch_norm=last_batch_norm, dropout=dropout, use_3d=use_3d,
                                        graph_norm=graph_norm, edge_features=edge_features,
                                        edge_hidden_dim=edge_hidden_dim))
        # mixing network
        self.mixing_network = nn.Linear(out_dim, out_dim)
        self.mixing_act = nn.LeakyReLU()

    def forward(self, g, h, e, snorm_n):
        h_in = h  # for residual connection

        if self.divide_input:
            h_cat = torch.cat(
                [tower(g, h[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower],
                       e, snorm_n)
                 for n_tower, tower in enumerate(self.towers)], dim=1)
        else:
            h_cat = torch.cat([tower(g, h, e, snorm_n) for tower in self.towers], dim=1)

        h_out = self.mixing_act(self.mixing_network(h_cat))

        if self.residual:
            h_out = h_in + h_out  # residual connection
        return h_out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)


class PNAOriginalSimple(nn.Module):
    def __init__(self, hidden_dim, last_layer_dim, target_dim, in_feat_dropout, dropout, last_batch_norm,
                 mid_batch_norm, propagation_depth, readout_aggregators, readout_hidden_dim, readout_layers,
                 aggregators, scalers, avg_d, residual, posttrans_layers, readout_batchnorm, batch_norm_momentum, **kwargs):
        super().__init__()

        self.node_gnn = PNAGNNSimple(hidden_dim=hidden_dim, last_layer_dim=last_layer_dim, last_batch_norm=last_batch_norm,
                                mid_batch_norm=mid_batch_norm, in_feat_dropout=in_feat_dropout, dropout=dropout,
                                aggregators=aggregators, scalers=scalers, residual=residual, avg_d=avg_d,
                                propagation_depth=propagation_depth, posttrans_layers=posttrans_layers)

        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, g):
        h = g.ndata['feat']
        g, h = self.node_gnn(g, h)

        readouts_to_cat = [dgl.readout_nodes(g, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class PNAGNNSimple(nn.Module):
    def __init__(self, hidden_dim, last_layer_dim, in_feat_dropout, dropout, residual, aggregators, scalers, avg_d,
                 last_batch_norm, mid_batch_norm, propagation_depth, posttrans_layers):
        super().__init__()

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)

        self.layers = nn.ModuleList(
            [PNASimpleLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                            last_batch_norm=last_batch_norm, mid_batch_norm=mid_batch_norm, residual=residual,
                            aggregators=aggregators,
                            scalers=scalers, avg_d=avg_d, posttrans_layers=posttrans_layers)
             for _ in range(propagation_depth - 1)])

        self.layers.append(PNASimpleLayer(in_dim=hidden_dim, out_dim=last_layer_dim, dropout=dropout,
                                          last_batch_norm=last_batch_norm, mid_batch_norm=mid_batch_norm,
                                          residual=residual, aggregators=aggregators, scalers=scalers,
                                          avg_d=avg_d, posttrans_layers=posttrans_layers))

        self.output = MLPReadout(last_layer_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, h):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, conv in enumerate(self.layers):
            h = conv(g, h)

        g.ndata['feat'] = h

        return g, h


class PNASimpleLayer(nn.Module):

    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, last_batch_norm, mid_batch_norm, residual,
                 posttrans_layers=1):
        super().__init__()

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        scalers = [SCALERS[scale] for scale in scalers]

        self.aggregators = aggregators
        self.scalers = scalers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual

        self.posttrans = MLP(in_dim=(len(aggregators) * len(scalers)) * in_dim, hidden_size=out_dim,
                             last_batch_norm=last_batch_norm,
                             mid_batch_norm=mid_batch_norm,
                             out_dim=out_dim, layers=posttrans_layers, mid_activation='relu',
                             last_activation='none')
        self.avg_d = avg_d
        self.dropout = nn.Dropout(p=dropout)

    def reduce_func(self, nodes):
        h = nodes.mailbox['m']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'feat': h}

    def forward(self, g, h):
        h_in = h
        g.ndata['feat'] = h

        # aggregation
        g.update_all(fn.copy_u('feat', 'm'), self.reduce_func)
        h = g.ndata['feat']

        # posttransformation
        h = self.posttrans(h)

        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)
