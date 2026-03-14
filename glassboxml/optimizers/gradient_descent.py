import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, params, grads):
        """
        params: dict containing model parameters (e.g., weights and bias)
        grads: dict containing gradients of the loss with respect to the parameters
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]

        return params