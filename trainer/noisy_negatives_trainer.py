import torch

from trainer.self_supervised_trainer import SelfSupervisedTrainer


class NoisyNegativesTrainer(SelfSupervisedTrainer):
    def __init__(self, **kwargs):
        super(NoisyNegativesTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        graph, info3d, noisy3d = tuple(batch)
        view2d = self.model(graph)
        view3d = self.model3d(info3d)
        loss = self.loss_func(view2d, view3d, nodes_per_graph=graph.batch_num_nodes())

        return loss, view2d, view3d
