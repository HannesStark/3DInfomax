import torch

from trainer.self_supervised_trainer import SelfSupervisedTrainer


class SelfSupervisedAlternatingTrainer(SelfSupervisedTrainer):
    def __init__(self, **kwargs):
        super(SelfSupervisedAlternatingTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        graph, info3d, *targets = tuple(batch)
        if self.optim_steps % 2 == 0:
            view2d = self.model(graph)
            with torch.no_grad():
                view3d = self.model3d(info3d)
            loss = self.loss_func(view2d, view3d, nodes_per_graph=graph.batch_num_nodes())
        else:
            with torch.no_grad():
                view2d = self.model(graph)
            view3d = self.model3d(info3d)
            loss = self.loss_func(view3d, view2d, nodes_per_graph=graph.batch_num_nodes())
        return loss, view2d, view3d
