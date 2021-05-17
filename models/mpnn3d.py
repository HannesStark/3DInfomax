import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl.function as fn


class MPNN3D(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, target_dim, n_layers: int = 4, **kwargs):
        super(MPNN3D, self).__init__()
        self.input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Linear(edge_dim, hidden_dim)
        self.mp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mp_layers.append(MPLayer(
                message_network=nn.Sequential(nn.Linear(3 * hidden_dim + 1, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, hidden_dim),
                                              nn.ReLU()),
                update_network=nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU()
                                             )))

        self.output = nn.Linear(2 * hidden_dim, target_dim)

    def forward(self, graph: dgl.DGLGraph):
        graph.apply_nodes(self.input_node_func)
        graph.apply_edges(self.input_edge_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        mean_nodes = dgl.mean_nodes(graph, 'f')
        max_nodes = dgl.max_nodes(graph, 'f')
        mean_max = torch.cat([mean_nodes, max_nodes], dim=-1)
        return self.output(mean_max)

    def input_node_func(self, nodes):
        return {'f': F.relu(self.input(nodes.data['f']))}

    def input_edge_func(self, edges):
        return {'w': F.relu(self.edge_input(edges.data['w']))}


class MPLayer(nn.Module):
    def __init__(self, message_network, update_network):
        super(MPLayer, self).__init__()
        self.message_network = message_network
        self.update_network = update_network

    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=fn.sum(msg='m', out='m_sum'))
        graph.apply_nodes(self.update_function)

    def message_function(self, edges):
        squared_distance = torch.sum((edges.src['x'] - edges.dst['x']) ** 2, dim=-1)[:, None]
        message_input = torch.cat([edges.src['f'], edges.dst['f'], edges.data['w'], squared_distance], dim=-1)
        message = self.message_network(message_input)
        return {'m': message}

    def update_function(self, nodes):
        return {'f': self.update_network(nodes.data['m_sum'] + nodes.data['f'])}
