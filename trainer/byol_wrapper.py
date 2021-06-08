import copy

import dgl
import torch

from torch import nn

from models import * # do not reomve
from models.base_layers import MLP


class BYOLwrapper(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self,
                 model_type,
                 model_parameters,
                 predictor_layers=1,
                 predictor_hidden_size=256,
                 predictor_batchnorm = False,
                 metric_dim=256,
                 ma_decay=0.99, #moving average decay
                 **kwargs):
        super(BYOLwrapper, self).__init__()
        self.student = globals()[model_type](**model_parameters, **kwargs)
        self.teacher = copy.deepcopy(self.student)
        self.predictor_layers = predictor_layers
        if predictor_layers > 0:
            self.predictor = MLP(in_dim=model_parameters['target_dim'], hidden_size=predictor_hidden_size,
                          mid_batch_norm=predictor_batchnorm, out_dim=metric_dim,
                          layers=predictor_layers)
        self.ma_decay = ma_decay
        for p in self.teacher.parameters():
            p.requires_grad = False

    def ma_teacher_update(self):
        for params_s, params_t in zip(self.student.parameters(), self.teacher.parameters()):
            params_t.data = params_t.data * self.ma_decay + params_s.data * (1. - self.ma_decay)

    def forward(self, graph: dgl.DGLGraph):
        graph_t = copy.deepcopy(graph)
        projection_s = self.student(graph)
        if self.predictor_layers >0:
            prediction_s = self.predictor(projection_s)
        else:
            prediction_s = projection_s

        with torch.no_grad():
            projection_t = self.teacher(graph_t)

        return prediction_s, projection_t
