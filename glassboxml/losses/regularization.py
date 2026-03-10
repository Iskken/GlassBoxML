import numpy as np

class Regularization:
    def __init__(self, lambda_=0):
        self.lambda_ = lambda_
        pass

    def loss(self, w):
        return NotImplementedError
    
    def gradient(self, w):
        return NotImplementedError

class L1Regularization(Regularization):
    def __init__(self, lambda_=0):
        self.lambda_ = lambda_
        pass

    def loss(self, w):
        return self.lambda_ * np.sum(np.abs(w))

    def gradient(self, w):
        return self.lambda_ * np.sign(w)

class L2Regularization(Regularization):
    def __init__(self, lambda_=0):
        self.lambda_ = lambda_
        pass

    def loss(self, w):
        return self.lambda_ * np.sum(w**2)

    def gradient(self, w):
        return 2 * self.lambda_ * w