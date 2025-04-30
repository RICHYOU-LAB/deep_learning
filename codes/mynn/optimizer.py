from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    def apply_weight_decay(self, param, decay, lr):

        return param * (1 - lr * decay)

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if getattr(layer, 'weight_decay', False):
                        layer.params[key] = self.apply_weight_decay(
                            layer.params[key],
                            getattr(layer, 'weight_decay_lambda', 0.0),
                            self.init_lr
                        )
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        def __init__(self, init_lr, model, mu, scheduler=None):
            super().__init__(init_lr, model, scheduler)
            self.mu = mu
            self.v = {}
            for layer in self.model.layers:
                if layer.optimizable:
                    for key in layer.params.keys():
                        self.v[(layer, key)] = np.zeros_like(layer.params[key])


    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    vel_key = (id(layer), key)

                    self.velocities[vel_key] = (
                            self.mu * self.velocities[vel_key]
                            - self.init_lr * layer.grads[key]
                    )

                    if getattr(layer, 'weight_decay', False):
                        layer.params[key] = self.apply_weight_decay(
                            layer.params[key],
                            getattr(layer, 'weight_decay_lambda', 0.0),
                            self.init_lr
                        )

                    layer.params[key] += self.velocities[vel_key]