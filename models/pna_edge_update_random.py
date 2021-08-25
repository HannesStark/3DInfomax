from typing import Dict, List, Union, Callable

import dgl
import torch

from torch import nn

from commons.mol_encoder import AtomEncoder, BondEncoder
from models.base_layers import MLP
from models.geomol_mpnn import GeomolMLP
from models.pna import PNA_AGGREGATORS, PNA_SCALERS
import torch.nn.functional as F


class PNARandomEdgeUpdate(nn.Module):
    def __init__(self, hidden_dim, target_dim, random_vec_dim, random_vec_std, aggregators: List[str],
                 scalers: List[str],
                 readout_aggregators: List[str], readout_batchnorm: bool = True, readout_hidden_dim=None,
                 readout_layers: int = 2, residual: bool = True, pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu", last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False, last_batch_norm: bool = False, propagation_depth: int = 5,
                 dropout: float = 0.0, posttrans_layers: int = 1, pretrans_layers: int = 1, batch_norm_momentum=0.1,
                 **kwargs):
        super(PNARandomEdgeUpdate, self).__init__()
        self.node_gnn = PNAGNNRandomEdgeUpdate(random_vec_dim=random_vec_dim, hidden_dim=hidden_dim,
                                               aggregators=aggregators,
                                               scalers=scalers, residual=residual,
                                               pairwise_distances=pairwise_distances,
                                               activation=activation, last_activation=last_activation,
                                               mid_batch_norm=mid_batch_norm,
                                               last_batch_norm=last_batch_norm, propagation_depth=propagation_depth,
                                               dropout=dropout,
                                               posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                                               batch_norm_momentum=batch_norm_momentum
                                               )
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph: dgl.DGLGraph):
        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        # rand_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        rand_x = rand_dist.sample([graph.ndata['feat'].size(0), self.random_vec_dim]).squeeze(-1).to(graph.device)
        rand_edge = rand_dist.sample([graph.edata['feat'].size(0), self.random_vec_dim]).squeeze(-1).to(graph.device)

        self.node_gnn(rand_x, rand_edge, graph)
        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class PNAGNNRandomEdgeUpdate(nn.Module):
    def __init__(self, random_vec_dim, hidden_dim, aggregators: List[str], scalers: List[str], n_model_confs=10,
                 residual: bool = True, pairwise_distances: bool = False, activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none", mid_batch_norm: bool = False,
                 last_batch_norm: bool = False, batch_norm_momentum=0.1, propagation_depth: int = 5,
                 dropout: float = 0.0, posttrans_layers: int = 1, pretrans_layers: int = 1, pretrain_mode=False,
                 **kwargs):
        super(PNAGNNRandomEdgeUpdate, self).__init__()
        self.mp_layers = nn.ModuleList()
        self.random_vec_dim = random_vec_dim
        self.n_model_confs = n_model_confs
        self.pretrain_mode = pretrain_mode

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)
        self.node_init = GeomolMLP(hidden_dim + random_vec_dim, hidden_dim, num_layers=2,
                                   batch_norm_momentum=batch_norm_momentum)
        self.edge_init = GeomolMLP(hidden_dim + random_vec_dim, hidden_dim, num_layers=2,
                                   batch_norm_momentum=batch_norm_momentum)
        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayerEdgeUpdate(in_dim=hidden_dim, out_dim=int(hidden_dim), in_dim_edges=hidden_dim,
                                   aggregators=aggregators,
                                   scalers=scalers, pairwise_distances=pairwise_distances, residual=residual,
                                   dropout=dropout,
                                   activation=activation, last_activation=last_activation,
                                   mid_batch_norm=mid_batch_norm,
                                   last_batch_norm=last_batch_norm, avg_d={"log": 1.0},
                                   posttrans_layers=posttrans_layers,
                                   pretrans_layers=pretrans_layers, batch_norm_momentum=batch_norm_momentum
                                   ),

            )


    def forward(self, rand_x, rand_edge, dgl_graph: dgl.DGLGraph, **kwargs):
        dgl_graph.ndata['feat'] = self.atom_encoder(dgl_graph.ndata['feat'])
        dgl_graph.edata['feat'] = self.bond_encoder(dgl_graph.edata['feat'])
        if self.pretrain_mode:
            n_atoms, hidden_dim = dgl_graph.ndata['feat'].size()
            graph_confs = []
            for i in range(self.n_model_confs):
                graph_confs.append(dgl_graph.clone())
            graph_confs = dgl.batch(graph_confs)
            n_all_atoms = graph_confs.number_of_nodes()
            n_all_edges = graph_confs.number_of_edges()
            dgl_graph = graph_confs
            rand_x = rand_x.view(n_all_atoms, -1)
            rand_edge = rand_edge.view(n_all_edges, -1)
        dgl_graph.ndata['feat'] = self.node_init(torch.cat([dgl_graph.ndata['feat'], rand_x], dim=-1))
        dgl_graph.edata['feat'] = self.edge_init(torch.cat([dgl_graph.edata['feat'], rand_edge], dim=-1))

        for mp_layer in self.mp_layers:
            mp_layer(dgl_graph)

        if self.pretrain_mode:
            feat = dgl_graph.ndata['feat'].view(n_atoms, -1, hidden_dim)
        else:
            feat = dgl_graph.ndata['feat']
        return feat, None


class PNALayerEdgeUpdate(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, in_dim_edges: int, aggregators: List[str], scalers: List[str],
                 activation: Union[Callable, str] = "relu", last_activation: Union[Callable, str] = "none",
                 dropout: float = 0.0, residual: bool = True, pairwise_distances: bool = False,
                 mid_batch_norm: bool = False, last_batch_norm: bool = False, batch_norm_momentum=0.1,
                 avg_d: Dict[str, float] = {"log": 1.0}, posttrans_layers: int = 2, pretrans_layers: int = 1, ):
        super(PNALayerEdgeUpdate, self).__init__()
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.edge = nn.Linear(in_dim, in_dim)
        self.node_in = nn.Linear(in_dim, in_dim, bias=False)
        self.node_out = nn.Linear(in_dim, in_dim, bias=False)

        self.pretrans = MLP(
            in_dim=in_dim,
            hidden_size=in_dim, out_dim=in_dim, mid_batch_norm=mid_batch_norm, last_batch_norm=last_batch_norm,
            layers=pretrans_layers, mid_activation=activation, dropout=dropout, last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum

        )
        self.edge_eps = nn.Parameter(torch.Tensor([0]))
        self.node_eps = nn.Parameter(torch.Tensor([0]))
        self.posttrans_1 = MLP(in_dim=in_dim, hidden_size=out_dim,
                               out_dim=out_dim, layers=posttrans_layers, mid_activation=activation,
                               last_activation=last_activation, dropout=dropout, mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm, batch_norm_momentum=batch_norm_momentum
                               )
        self.posttrans_2 = MLP(in_dim=(len(self.aggregators) * len(self.scalers)) * in_dim, hidden_size=out_dim,
                               out_dim=out_dim, layers=posttrans_layers, mid_activation=activation,
                               last_activation=last_activation, dropout=dropout, mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm, batch_norm_momentum=batch_norm_momentum
                               )

    def forward(self, g):
        h = g.ndata['feat']
        h_in = h
        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        # post-transformation
        h = (1 + self.node_eps) * h_in + self.posttrans_2(g.ndata['feat'])

        g.ndata['feat'] = h

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {'e': edges.data['feat']}

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data['feat']
        h = self.posttrans_1(nodes.mailbox['e'])
        D = h.shape[-2]
        #h_to_cat = [aggr(h=h, h_in=h_in) for aggr in self.aggregators]
        #h = torch.cat(h_to_cat, dim=-1)
#
        #if len(self.scalers) > 1:
        #    h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=-1)

        return {'feat': h}

    def pretrans_edges(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).
        """
        edge = self.edge(edges.data['feat'])
        node_in = self.node_in(edges.src['feat'])
        node_out = self.node_out(edges.dst['feat'])

        out = F.relu(edge + node_in + node_out)
        return {'feat': (1 + self.edge_eps) * edges.data['feat'] + self.pretrans(out)}
