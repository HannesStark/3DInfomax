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


class OGBGNN(nn.Module):

    def __init__(self, target_dim = 1, num_layers = 5, hidden_dim = 300, gnn_type = 'gin',
                 virtual_node = True, residual = False, dropout = 0, JK = "last",
                 graph_pooling = "sum", batch_norm_momentum=0.1, **kwargs):
        '''
            target_dim (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(OGBGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.JK = JK
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.node_gnn = GNN_node_Virtualnode(num_layers, hidden_dim, JK = JK,
                                                 dropout = dropout,
                                                 residual = residual,
                                                 gnn_type = gnn_type, batch_norm_momentum=batch_norm_momentum)
        else:
            self.node_gnn = GNN_node(num_layers, hidden_dim, JK = JK, dropout = dropout,
                                     residual = residual, gnn_type = gnn_type, batch_norm_momentum=batch_norm_momentum)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = SumPooling()
        elif self.graph_pooling == "mean":
            self.pool = AvgPooling()
        elif self.graph_pooling == "max":
            self.pool = MaxPooling
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttentionPooling(
                gate_nn = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim),
                                        nn.BatchNorm1d(2*hidden_dim, momentum=batch_norm_momentum),
                                        nn.ReLU(),
                                        nn.Linear(2*hidden_dim, 1)))

        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(hidden_dim, n_iters = 2, n_layers = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2*self.hidden_dim, self.target_dim)
        else:
            self.graph_pred_linear = nn.Linear(self.hidden_dim, self.target_dim)

    def forward(self, g):
        x = g.ndata['feat']
        edge_attr = g.edata['feat']
        h_node = self.node_gnn(g, x, edge_attr)

        h_graph = self.pool(g, h_node)
        output = self.graph_pred_linear(h_graph)

        return output


### GIN convolution along the graph structure
class GINConv(nn.Module):
    def __init__(self, hidden_dim, batch_norm_momentum=0.1):
        '''
            hidden_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim, momentum=batch_norm_momentum),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = hidden_dim)

    def forward(self, g, x, edge_attr):
        with g.local_scope():
            edge_embedding = self.bond_encoder(edge_attr)
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
        self.bond_encoder = BondEncoder(emb_dim = hidden_dim)

    def forward(self, g, x, edge_attr):
        with g.local_scope():
            x = self.linear(x)
            edge_embedding = self.bond_encoder(edge_attr)

            # Molecular graphs are undirected
            # g.out_degrees() is the same as g.in_degrees()
            degs = (g.out_degrees().float() + 1).to(x.device)
            norm = torch.pow(degs, -0.5).unsqueeze(-1)                # (N, 1)
            g.ndata['norm'] = norm
            g.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))

            g.ndata['x'] = x
            g.apply_edges(fn.copy_u('x', 'm'))
            g.edata['m'] = g.edata['norm'] * F.relu(g.edata['m'] + edge_embedding)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'new_x'))
            out = g.ndata['new_x'] + F.relu(x + self.root_emb.weight) * 1. / degs.view(-1, 1)

            return out

### GNN to generate node embedding
class GNN_node(nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, hidden_dim, dropout = 0.5, JK = "last", residual = False, gnn_type = 'gin', batch_norm_momentum=0.1):
        '''
            num_layers (int): number of GNN message passing layers
            hidden_dim (int): node embedding dimensionality
        '''

        super(GNN_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(hidden_dim)

        ###List of GNNs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(hidden_dim, batch_norm_momentum))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, batch_norm_momentum))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, momentum=batch_norm_momentum))

    def forward(self, g, x, edge_attr):
        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):

            h = self.convs[layer](g, h_list[layer], edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training = self.training)

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

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, hidden_dim, dropout = 0.5, JK = "last", residual = False, gnn_type = 'gin', batch_norm_momentum=0.1):
        '''
            num_layers (int): number of GNN message passing layers
            hidden_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.atom_encoder = AtomEncoder(hidden_dim)

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
                self.convs.append(GINConv(hidden_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, momentum=batch_norm_momentum))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                           nn.BatchNorm1d(hidden_dim, momentum=batch_norm_momentum),
                                                           nn.ReLU(),
                                                           nn.Linear(hidden_dim, hidden_dim),
                                                           nn.BatchNorm1d(hidden_dim, momentum=batch_norm_momentum),
                                                           nn.ReLU()))
        self.pool = SumPooling()

    def forward(self, g, x, edge_attr):
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(g.batch_size).to(x.dtype).to(x.device))

        h_list = [self.atom_encoder(x)]
        batch_id = dgl.broadcast_nodes(g, torch.arange(g.batch_size).to(x.device))
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch_id]

            ### Message passing among graph nodes
            h = self.convs[layer](g, h_list[layer], edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training = self.training)

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
                        virtualnode_embedding_temp, self.dropout, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(
                        virtualnode_embedding_temp, self.dropout, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation