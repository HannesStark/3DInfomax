import os
import torch.nn.functional as F
import dgl
import numpy as np
import torch
import torch_geometric

from ogb.graphproppred import GraphPropPredDataset, DglGraphPropPredDataset
from torch.utils.data import Subset


class OGBGDatasetExtension(GraphPropPredDataset):
    def __init__(self, return_types, name, device, root='dataset', meta_dict=None, num_freq=10):
        super(OGBGDatasetExtension, self).__init__(name=name, root=root, meta_dict=meta_dict)
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''
        self.return_types = return_types
        self.dgl_graphs = {}
        self.san_graphs = {}
        self.pos_enc = {}
        self.num_freq = num_freq

        self.device = device
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        data = []
        for return_type in self.return_types:
            data.append(self.data_by_type(idx, return_type))
        return tuple(data)

    def data_by_type(self, idx, return_type):
        if return_type == 'dgl_graph':
            return self.get_graph(idx).to(self.device)
        elif return_type == 'raw_features':
            return torch.tensor(self.graphs[idx]['node_feat']).to(self.device)
        elif return_type == 'targets':
            return self.labels[idx].to(self.device)
        elif return_type == 'pytorch_geometric_graph':
            graph_info = self.graphs[idx]
            return torch_geometric.data.Data(z=torch.from_numpy(graph_info['node_feat']).to(self.device),
                                             edge_attr=torch.from_numpy(graph_info['edge_feat']).to(self.device),
                                             edge_index=torch.from_numpy(graph_info['edge_index']).to(self.device),
                                             num_nodes=graph_info['num_nodes'])
        elif return_type == 'positional_encoding':
            eig_vals, eig_vecs = self.get_pos_enc(idx)
            eig_vals = eig_vals.to(self.device)
            eig_vecs = eig_vecs.to(self.device)
            sign_flip = torch.rand(eig_vals.shape[0], device=self.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eig_vals = eig_vals.unsqueeze(0).repeat(eig_vecs.shape[0], 1)
            return torch.stack([eig_vals, eig_vecs], dim=-1)
        if return_type == 'san_graph':
            g = self.get_san_graph(idx).to(self.device)

            eig_vals, eig_vecs = self.get_pos_enc(idx)
            eig_vals = eig_vals.to(self.device)
            eig_vecs = eig_vecs.to(self.device)
            sign_flip = torch.rand(eig_vals.shape[0], device=self.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eig_vals = eig_vals.unsqueeze(0).repeat(eig_vecs.shape[0], 1)
            g.ndata['pos_enc'] = torch.stack([eig_vals, eig_vecs], dim=-1)
            return g

    def get_san_graph(self, idx):
        if idx in self.san_graphs:
            return self.san_graphs[idx]
        else:
            graph_info = self.graphs[idx]
            n_atoms = graph_info['num_nodes']
            src = torch.repeat_interleave(torch.arange(n_atoms, device=self.device), n_atoms - 1)
            dst = torch.cat([torch.cat(
                [torch.arange(n_atoms, device=self.device)[:idx], torch.arange(n_atoms, device=self.device)[idx + 1:]])
                for idx in range(n_atoms)])  # without self loops
            g = dgl.graph((src, dst), device=self.device)
            g.ndata['feat'] = torch.from_numpy(graph_info['node_feat']).to(self.device)

            e_features = torch.from_numpy(graph_info['edge_feat']).to(self.device)
            g.edata['feat'] = torch.zeros(g.number_of_edges(), e_features.shape[1], dtype=torch.long,
                                          device=self.device)
            g.edata['real'] = torch.zeros(g.number_of_edges(), dtype=torch.long, device=self.device)
            g.edges[graph_info['edge_index'][0], graph_info['edge_index'][1]].data['feat'] = e_features
            g.edges[graph_info['edge_index'][0], graph_info['edge_index'][1]].data['real'] = torch.ones(
                e_features.shape[0],
                dtype=torch.long,
                device=self.device)  # This indicates real edges
            self.san_graphs[idx] = g.cpu()
            return g

    def get_graph(self, idx):
        if idx in self.dgl_graphs:
            return self.dgl_graphs[idx]
        else:
            graph_info = self.graphs[idx]
            g = dgl.graph((graph_info['edge_index'][0], graph_info['edge_index'][1]), num_nodes=graph_info['num_nodes'])
            g.ndata['feat'] = torch.from_numpy(graph_info['node_feat'])
            g.edata['feat'] = torch.from_numpy(graph_info['edge_feat'])
            return g

    def get_pos_enc(self, idx):
        if idx in self.pos_enc:
            return self.pos_enc[idx]
        else:
            graph_info = self.graphs[idx]
            edge_index = graph_info['edge_index']
            n_atoms = graph_info['num_nodes']
            sparse = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), size=(n_atoms, n_atoms),
                                             device=self.device)
            A = sparse.to_dense()
            A += torch.eye(n_atoms, device=self.device)
            D = A.sum(dim=0)
            D_mat = torch.diag(D)
            L = D_mat - A
            D += (D == 0)
            N = D ** -0.5
            L_sym = torch.eye(n_atoms, device=self.device) - N * L * N

            eig_vals, eig_vecs = torch.linalg.eigh(L_sym)
            idx = eig_vals.argsort()[0: self.num_freq]  # Keep up to the maximum desired number of frequencies
            eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]

            # Sort, normalize and pad EigenVectors
            eig_vecs = eig_vecs[:, eig_vals.argsort()]  # increasing order
            eig_vecs = F.normalize(eig_vecs, p=2, dim=1, eps=1e-12, out=None)
            if n_atoms < self.num_freq:
                eig_vecs = F.pad(eig_vecs, (0, self.num_freq - n_atoms), value=float('nan'))
                eig_vals = F.pad(eig_vals, (0, self.num_freq - n_atoms), value=float('nan'))
            self.pos_enc[idx] = (eig_vals.cpu(), eig_vecs.cpu())
            return self.pos_enc[idx]
