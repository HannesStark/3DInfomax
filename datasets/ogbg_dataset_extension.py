import dgl
import numpy as np
import torch

from ogb.graphproppred import  GraphPropPredDataset, DglGraphPropPredDataset
from torch.utils.data import Subset


class OGBGDatsetExtension(GraphPropPredDataset):
    def __init__(self, return_types, name, device, root='dataset', meta_dict=None):
        super(OGBGDatsetExtension, self).__init__(name=name, root=root, meta_dict=meta_dict)
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''
        self.return_types = return_types
        self.dgl_graphs = {}
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
            return self.graphs[idx]['node_feat'].to(self.device)
        elif return_type == 'targets':
            return self.labels[idx].to(self.device)

    def get_graph(self, idx):
        if idx in self.dgl_graphs:
            return self.dgl_graphs[idx]
        else:
            graph_info = self.graphs[idx]
            g = dgl.graph((graph_info['edge_index'][0], graph_info['edge_index'][1]))
            g.ndata['feat'] = torch.tensor(graph_info['node_feat'])
            g.edata['feat'] = torch.tensor(graph_info['edge_feat'])
            return g
