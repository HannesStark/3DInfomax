from commons.utils import tensorboard_gradient_magnitude
from trainer.lr_schedulers import WarmUpWrapper
from trainer.self_supervised_trainer import SelfSupervisedTrainer


class BYOLTrainer(SelfSupervisedTrainer):
    def __init__(self, **kwargs):
        super(BYOLTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        graph, info3d, *targets = tuple(batch)
        prediction2d_student, projection2d_teacher = self.model(graph)  # foward the rest of the batch to the model
        prediction3d_student, projection3d_teacher = self.model3d(info3d)

        nodes_per_graph = graph.batch_num_nodes()
        loss2d_student = self.loss_func(prediction2d_student, projection3d_teacher, nodes_per_graph=nodes_per_graph)
        loss3d_student = self.loss_func(projection2d_teacher, prediction3d_student, nodes_per_graph=nodes_per_graph)
        loss = loss2d_student + loss3d_student
        return loss, prediction2d_student, prediction3d_student

    def after_optim_step(self):
        self.model.ma_teacher_update()
        if self.optim_steps % self.args.log_iterations == 0:
            tensorboard_gradient_magnitude(self.optim, self.writer, self.optim_steps)
        if self.lr_scheduler != None and (self.scheduler_step_per_batch or (isinstance(self.lr_scheduler,
                                                                                       WarmUpWrapper) and self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
            self.lr_scheduler.step()
