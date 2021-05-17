from torch.optim.lr_scheduler import *
import numpy as np


class WarmUpWrapper:
    "Optim wrapper that implements lr."

    def __init__(self, optimizer, wrapped_scheduler, warmup_steps, interpolation='cosine', **kwargs):
        self.optim = optimizer
        self._step = 0
        self.interpolation = interpolation
        self.warmup_steps = warmup_steps
        self.wrapped_scheduler = globals()[wrapped_scheduler](self.optim, **kwargs)
        self.start_lrs = []
        for p in self.optim.param_groups:
            self.start_lrs.append(p['lr'])

    def step(self, metrics=None):
        "Update parameters and lr"
        if self._step <= self.warmup_steps:
            for i, p in enumerate(self.optim.param_groups):
                # interpolate between 0 and the final starting learning rate
                if self.interpolation == 'linear':
                    p['lr'] = self.start_lrs[i] * (self._step / self.warmup_steps)
                elif self.interpolation == 'cosine':
                    p['lr'] = self.start_lrs[i] * ((-np.cos((np.pi) * (self._step / self.warmup_steps)) + 1) * 0.5)
                else:
                    raise ValueError('interpolation not implemented:', self.interpolation)
            self.optim.step()
        else:
            if metrics != None:
                self.wrapped_scheduler.step(metrics=metrics)
            else:
                self.wrapped_scheduler.step()
        self._step += 1

    def state_dict(self):
        """Returns the state of the warmup_steps scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optim.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optim'}
        state_dict['wrapped_scheduler'] = self.wrapped_scheduler.state_dict()  # overwrite with the state dict
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the warmup_steps scheduler's state.
        Arguments:
            state_dict (dict): warmup_steps scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        wrapped_scheduler_state_dict = state_dict['wrapped_scheduler']
        del state_dict['wrapped_scheduler']
        self.wrapped_scheduler.load_state_dict(wrapped_scheduler_state_dict)
        self.__dict__.update(state_dict)
