import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from models.base_layers import MLP


class EGNNDistEmbedding(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, target_dim, batch_norm=False, dropout=0.0, propagation_depth: int = 4,
                 mid_activation: str = 'SiLU', ** kwargs):
        super(EGNNDistEmbedding, self).__init__()
        self.input = MLP(
            in_dim=node_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=1,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )
        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(EGCLayer(node_dim, hidden_dim=hidden_dim, batch_norm=batch_norm, dropout=dropout,
                                           mid_activation=mid_activation))

        self.node_wise_output_network = MLP(
            in_dim=hidden_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )
        self.output_network = MLP(in_dim=2*hidden_dim, hidden_size=hidden_dim, mid_activation=mid_activation,
                                  mid_batch_norm=batch_norm, out_dim=target_dim,
                                  layers=2)

    def forward(self, graph: dgl.DGLGraph):
        graph.apply_nodes(self.input_node_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        graph.apply_nodes(self.output_node_func)
        sum_nodes = dgl.sum_nodes(graph, 'feat')
        max_nodes = dgl.max_nodes(graph, 'feat')
        sum_max = torch.cat([sum_nodes, max_nodes], dim=-1)
        mol_property = self.output_network(sum_max)
        return mol_property

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_node_func(self, nodes):
        return {'feat': F.silu(self.input(nodes.data['feat']))}

    def input_edge_func(self, edges):
        return {'feat': F.silu(self.edge_input(edges.data['feat']))}


class EGCLayer(nn.Module):
    def __init__(self, node_dim, hidden_dim, batch_norm, dropout, mid_activation):
        super(EGCLayer, self).__init__()
        self.message_network = MLP(
            in_dim=hidden_dim * 2 + 6,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation=mid_activation,
        )

        self.update_network = MLP(
            in_dim=hidden_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation='None',
        )

        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=fn.sum(msg='m', out='m_sum'),
                         apply_node_func=self.update_function)

    def message_function(self, edges):
        message_input = torch.cat(
            [edges.src['feat'], edges.dst['feat'], edges.data['d_rbf']], dim=-1)
        message = self.message_network(message_input)
        edge_weight = torch.sigmoid(message)
        return {'m': message * edge_weight}

    def update_function(self, nodes):
        h = nodes.data['feat']
        input = torch.cat([nodes.data['m_sum'] + nodes.data['feat']], dim=-1)
        h_new = self.update_network(input)
        output = h_new + h
        return {'feat': output}
