from trainer.self_supervised_trainer import SelfSupervisedTrainer


class BYOLTrainer(SelfSupervisedTrainer):
    def __init__(self, **kwargs):
        super(BYOLTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        graph, info3d = tuple(batch)
        prediction2d_student, projection2d_teacher = self.model(graph)  # foward the rest of the batch to the model
        prediction3d_student, projection3d_teacher = self.model3d(info3d)

        nodes_per_graph = graph.batch_num_nodes()
        loss2d_student = self.loss_func(prediction2d_student, projection3d_teacher, nodes_per_graph=nodes_per_graph)
        loss3d_student = self.loss_func(projection2d_teacher, prediction3d_student, nodes_per_graph=nodes_per_graph)
        loss = loss2d_student + loss3d_student
        return loss, prediction2d_student, prediction3d_student

    def after_optim_step(self):
        self.model.ma_teacher_update()
