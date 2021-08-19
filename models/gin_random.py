import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import SumPooling
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

import torch
import torch.nn as nn

from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling, Set2Set


class OGBGNNRandom(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, hidden_dim=300, gnn_type='gin',
                 virtual_node=True, residual=False, dropout=0, JK="last",
                 graph_pooling="sum", random_vec_dim=10, random_vec_std=1.0, **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(OGBGNNRandom, self).__init__()
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        self.num_layers = num_layers
        self.dropout = dropout
        self.JK = JK
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.node_gnn = GNN_node_VirtualnodeRandom(num_layers, hidden_dim, JK=JK,
                                                       dropout=dropout,
                                                       residual=residual,
                                                       gnn_type=gnn_type, random_vec_dim=random_vec_dim)
        else:
            self.node_gnn = GNN_nodeRandom(num_layers, hidden_dim, JK=JK, dropout=dropout,
                                           residual=residual, gnn_type=gnn_type, random_vec_dim=random_vec_dim)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = SumPooling()
        elif self.graph_pooling == "mean":
            self.pool = AvgPooling()
        elif self.graph_pooling == "max":
            self.pool = MaxPooling
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttentionPooling(
                gate_nn=nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim),
                                      nn.BatchNorm1d(2 * hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(2 * hidden_dim, 1)))

        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(hidden_dim, n_iters=2, n_layers=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2 * self.hidden_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.hidden_dim, self.num_tasks)

    def forward(self, g):
        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        # rand_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        rand_x = rand_dist.sample([g.ndata['feat'].size(0), self.random_vec_dim]).squeeze(-1).to(g.device)
        rand_edge = rand_dist.sample([g.edata['feat'].size(0), self.random_vec_dim]).squeeze(-1).to(g.device)

        x = g.ndata['feat']
        edge_attr = g.edata['feat']
        h_node, _ = self.node_gnn(rand_x=rand_x, rand_edge=rand_edge, dgl_graph=g, x=x, edge_attr=edge_attr)

        h_graph = self.pool(g, h_node)
        output = self.graph_pred_linear(h_graph)

        return output


### GIN convolution along the graph structure
class GINConvRandom(nn.Module):
    def __init__(self, hidden_dim, random_vec_dim):
        '''
            hidden_dim (int): node embedding dimensionality
        '''

        super(GINConvRandom, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=hidden_dim - random_vec_dim)

    def forward(self, g: dgl.DGLGraph, x, edge_attr, rand_edge):
        with g.local_scope():
            edge_embedding = self.bond_encoder(edge_attr)
            edge_embedding = torch.cat([edge_embedding, rand_edge], dim=-1)
            g.ndata['x'] = x
            g.apply_edges(fn.copy_u('x', 'm'))
            g.edata['m'] = F.relu(g.edata['m'] + edge_embedding)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'new_x'))
            out = self.mlp((1 + self.eps) * x + g.ndata['new_x'])

            return out


### GCN convolution along the graph structure
class GCNConv(nn.Module):
    def __init__(self, hidden_dim):
        '''
            hidden_dim (int): node embedding dimensionality
        '''

        super(GCNConv, self).__init__()

        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.root_emb = nn.Embedding(1, hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, g: dgl.DGLGraph, x, edge_attr):
        with g.local_scope():
            x = self.linear(x)
            edge_embedding = self.bond_encoder(edge_attr)

            # Molecular graphs are undirected
            # g.out_degrees() is the same as g.in_degrees()
            degs = (g.out_degrees().float() + 1).to(x.device)
            norm = torch.pow(degs, -0.5).unsqueeze(-1)  # (N, 1)
            g.ndata['norm'] = norm
            g.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))

            g.ndata['x'] = x
            g.apply_edges(fn.copy_u('x', 'm'))
            g.edata['m'] = g.edata['norm'] * F.relu(g.edata['m'] + edge_embedding)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'new_x'))
            out = g.ndata['new_x'] + F.relu(x + self.root_emb.weight) * 1. / degs.view(-1, 1)

            return out


### GNN to generate node embedding
class GNN_nodeRandom(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, hidden_dim, n_model_confs=10, dropout=0.5, JK="last", residual=False, gnn_type='gin',
                 random_vec_dim=10, pretrain_mode=False):
        '''
            num_layers (int): number of GNN message passing layers
            hidden_dim (int): node embedding dimensionality
        '''

        super(GNN_nodeRandom, self).__init__()
        self.num_layers = num_layers
        self.random_vec_dim = random_vec_dim
        self.dropout = dropout
        self.JK = JK
        self.n_model_confs = n_model_confs
        self.pretrain_mode = pretrain_mode
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(hidden_dim - random_vec_dim)

        ###List of GNNs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConvRandom(hidden_dim, random_vec_dim=random_vec_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, rand_x, rand_edge, dgl_graph, x, edge_attr, **kwargs):
        ### computing input node embedding
        dgl_graph.ndata['feat'] = self.atom_encoder(dgl_graph.ndata['feat'])
        if self.pretrain_mode:
            n_atoms, small_hidden_dim = dgl_graph.ndata['feat'].size()
            graph_confs = []
            for i in range(self.n_model_confs):
                graph_confs.append(dgl_graph.clone())
            graph_confs = dgl.batch(graph_confs)
            n_all_atoms = graph_confs.number_of_nodes()
            n_all_edges = graph_confs.number_of_edges()
            dgl_graph = graph_confs
            rand_x = rand_x.view(n_all_atoms, -1)
            rand_edge = rand_edge.view(n_all_edges, -1)
        dgl_graph.ndata['feat'] = torch.cat([dgl_graph.ndata['feat'], rand_x], dim=-1)
        g = dgl_graph
        x = dgl_graph.ndata['feat']

        h_list = [x]
        for layer in range(self.num_layers):

            h = self.convs[layer](g, h_list[layer], edge_attr=dgl_graph.edata['feat'].clone(), rand_edge=rand_edge)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        if self.pretrain_mode:
            node_representation = node_representation.view(n_atoms, -1, small_hidden_dim + self.random_vec_dim)
        else:
            node_representation = node_representation
        return node_representation, None


### Virtual GNN to generate node embedding
class GNN_node_VirtualnodeRandom(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, hidden_dim, dropout=0.5, JK="last", residual=False, gnn_type='gin',
                 pretrain_mode=False, random_vec_dim=10, n_model_confs=10):
        '''
            num_layers (int): number of GNN message passing layers
            hidden_dim (int): node embedding dimensionality
        '''

        super(GNN_node_VirtualnodeRandom, self).__init__()
        self.pretrain_mode = pretrain_mode
        self.n_model_confs = n_model_confs
        self.num_layers = num_layers
        self.dropout = dropout
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.random_vec_dim = random_vec_dim
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim - self.random_vec_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConvRandom(hidden_dim, random_vec_dim=random_vec_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                           nn.BatchNorm1d(hidden_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(hidden_dim, hidden_dim),
                                                           nn.BatchNorm1d(hidden_dim),
                                                           nn.ReLU()))
        self.pool = SumPooling()

    def forward(self, rand_x, rand_edge, dgl_graph, x, edge_attr, **kwargs):
        dgl_graph.ndata['feat'] = self.atom_encoder(dgl_graph.ndata['feat'])
        if self.pretrain_mode:
            n_atoms, small_hidden_dim = dgl_graph.ndata['feat'].size()
            graph_confs = []
            for i in range(self.n_model_confs):
                graph_confs.append(dgl_graph.clone())
            graph_confs = dgl.batch(graph_confs)
            n_all_atoms = graph_confs.number_of_nodes()
            n_all_edges = graph_confs.number_of_edges()
            dgl_graph = graph_confs
            rand_x = rand_x.view(n_all_atoms, -1)
            rand_edge = rand_edge.view(n_all_edges, -1)
        dgl_graph.ndata['feat'] = torch.cat([dgl_graph.ndata['feat'], rand_x], dim=-1)
        ### virtual node embeddings for graphs
        g = dgl_graph
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(g.batch_size).to(x.dtype).to(x.device))
        x = dgl_graph.ndata['feat']
        h_list = [x]
        batch_id = dgl.broadcast_nodes(g, torch.arange(g.batch_size).to(x.device))
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch_id]

            ### Message passing among graph nodes
            h = self.convs[layer](g, h_list[layer], dgl_graph.edata['feat'], rand_edge=rand_edge)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = self.pool(g, h_list[layer]) + virtualnode_embedding
                ### transform virtual nodes using MLP
                virtualnode_embedding_temp = self.mlp_virtualnode_list[layer](
                    virtualnode_embedding_temp)

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        virtualnode_embedding_temp, self.dropout, training=self.training)
                else:
                    virtualnode_embedding = F.dropout(
                        virtualnode_embedding_temp, self.dropout, training=self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        if self.pretrain_mode:
            node_representation = node_representation.view(n_atoms, -1, small_hidden_dim + self.random_vec_dim)
        else:
            node_representation = node_representation
        return node_representation, None
