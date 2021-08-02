import dgl
import torch
import torch.nn as nn
import dgl.function as fn

from commons.mol_encoder import AtomEncoder, BondEncoder
from models.base_layers import MLP, MLPReadout
from models.pna_original import GRU


class MPNN(nn.Module):
    """
    Message Passing Neural Network
    """

    def __init__(self, hidden_dim, last_layer_dim, target_dim, in_feat_dropout, dropout, last_batch_norm,
                 mid_batch_norm, propagation_depth, readout_aggregators, residual, posttrans_layers, pretrans_layers,
                 device, edge_hidden_dim, graph_norm, aggregation, distance_net_dim=False, gru_enable=False,
                 edge_feat=True,
                 **kwargs):
        super().__init__()

        self.gnn = MPNNGNN(hidden_dim=hidden_dim, last_layer_dim=last_layer_dim, last_batch_norm=last_batch_norm,
                           mid_batch_norm=mid_batch_norm, in_feat_dropout=in_feat_dropout, dropout=dropout,
                           aggregation=aggregation, residual=residual, propagation_depth=propagation_depth,
                           distance_net_dim=distance_net_dim, posttrans_layers=posttrans_layers, device=device,
                           pretrans_layers=pretrans_layers, gru_enable=gru_enable, edge_hidden_dim=edge_hidden_dim,
                           edge_feat=edge_feat, graph_norm=graph_norm)

        self.readout_aggregators = readout_aggregators
        self.output = MLPReadout(last_layer_dim * len(self.readout_aggregators), target_dim)

    def forward(self, g, snorm_n):
        h = g.ndata['feat']
        e = g.edata['feat']
        g, h = self.gnn(g, h, e, snorm_n)

        readouts_to_cat = [dgl.readout_nodes(g, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class MPNNGNN(nn.Module):
    def __init__(self, hidden_dim, last_layer_dim, in_feat_dropout, dropout, propagation_depth, graph_norm,
                 distance_net_dim, mid_batch_norm, last_batch_norm, residual, edge_feat, edge_hidden_dim,
                 pretrans_layers, aggregation, posttrans_layers, gru_enable, device):
        super().__init__()
        self.gru_enable = gru_enable
        self.edge_feat = edge_feat

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = AtomEncoder(hidden_dim)
        if self.edge_feat:
            self.embedding_e = BondEncoder(edge_hidden_dim)

        self.layers = nn.ModuleList([MPLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                                             graph_norm=graph_norm, mid_batch_norm=mid_batch_norm,
                                             last_batch_norm=last_batch_norm, residual=residual,
                                             edge_features=edge_feat, edge_hidden_dim=edge_hidden_dim,
                                             aggregation=aggregation, pretrans_layers=pretrans_layers,
                                             posttrans_layers=posttrans_layers, distance_net_dim=distance_net_dim) for _
                                     in range(propagation_depth - 1)])
        self.layers.append(MPLayer(in_dim=hidden_dim, out_dim=last_layer_dim, dropout=dropout, graph_norm=graph_norm,
                                   mid_batch_norm=mid_batch_norm, last_batch_norm=last_batch_norm, residual=residual,
                                   aggregation=aggregation, edge_features=edge_feat, edge_hidden_dim=edge_hidden_dim,
                                   pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers,
                                   distance_net_dim=distance_net_dim))

        if self.gru_enable:
            self.gru = GRU(hidden_dim, hidden_dim, device)

    def forward(self, g, h, e, snorm_n):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.edge_feat:
            e = self.embedding_e(e)

        for i, mp_layer in enumerate(self.layers):
            h_t = mp_layer(g, h, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t

        g.ndata['feat'] = h
        return g, h


class MPLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, graph_norm, mid_batch_norm, last_batch_norm, distance_net_dim=0,
                 aggregation='sum', pretrans_layers=1, posttrans_layers=1, residual=False, edge_features=False,
                 edge_hidden_dim=0):

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregation = aggregation
        self.edge_features = edge_features
        self.residual = residual
        self.distance_net_dim = distance_net_dim
        if self.distance_net_dim > 0:
            self.distance_net = MLP(in_dim=1, out_dim=distance_net_dim, layers=1, last_activation='relu')
        if in_dim != out_dim:
            self.residual = False
        self.dropout = nn.Dropout(dropout)
        self.graph_norm = graph_norm
        self.edge_features = edge_features
        self.pretrans = MLP(in_dim=2 * in_dim + (edge_hidden_dim if edge_features else 0) + distance_net_dim,
                            hidden_size=in_dim,
                            out_dim=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')
        self.posttrans = MLP(in_dim=2 * in_dim, hidden_size=out_dim, mid_batch_norm=mid_batch_norm,
                             last_batch_norm=last_batch_norm, out_dim=out_dim, layers=posttrans_layers,
                             mid_activation='relu', last_activation='none')

    def forward(self, g, h, e, snorm_n):
        h_in = h  # for residual connection
        g.ndata['feat'] = h
        if self.edge_features:  # add the edges information only if edge_features = True
            g.edata['feat'] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        if self.aggregation == 'sum':
            g.update_all(self.message_func, fn.sum('e', 'feat'))
        elif self.aggregation == 'mean':
            g.update_all(self.message_func, fn.mean('e', 'feat'))
        h = torch.cat([h, g.ndata['feat']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph normalization
        if self.graph_norm:
            h = h * snorm_n
        h_out = self.dropout(h)

        if self.residual:
            h_out = h_in + h_out  # residual connection
        return h_out

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1)
        else:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)
        if self.distance_net_dim > 0:
            distances = torch.norm((edges.src['x'] - edges.dst['x']), dim=-1)[:, None]
            distances = self.distance_net(distances)
            z2 = torch.cat([z2, distances], dim=1)
        return {'e': self.pretrans(z2)}

    def message_func(self, edges):
        return {'e': edges.data['e']}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['feat'])
