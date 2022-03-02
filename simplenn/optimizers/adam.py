from simplenn.layer import Layer
from simplenn.optimizers.optimizer import Optimizer
from typing import List
from typing import Union

import numpy as np


class Adam(Optimizer):
    """
    Adaptive Momentum Optimizer
    Per-parameter learning rate with cache (as RMSProp).
    Maintains a first and second moment of the gradient.
    First moment is the decaying mean gradient (controlled by b1).
    The estimation the mean of the gradient, uses a running average
    of gradients (similar to SGD).

    Second moment is the decaying variance (controlled by b2).
    The estimation of the variance of the gradient, uses a running
    sum of gradients (similar to RMSProp)


    Args:
        lr (float, optional): learning rate. Defaults to 0.1.
        decay (Union[float, None], optional): factor of linear decay per iteration
            to learning rate. Defaults to None.
        eps (float, optional): Epsilon. Avoids division by zero. Defaults to 1e-7 .
        b1 (float, optional): Factor for decaying moving average of partial
            dertivates how much momentum (weight of previous gradient) to include.
            Defaults to 0.9 .
        b2 (float, optional): Factor for decaying variance of partial
            dertivates how much momentum (weight of previous gradient) to include.
            Defaults to 0.999 .
    """

    def __init__(self, lr: float = 0.1, decay: Union[float, None] = None, eps: Union[float, None] = 1e-7, b1: float = 0.9, b2: float = 0.999) -> None:
        super().__init__()
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.b1 = b1
        self.b2 = b2

    def _get_cache(self, layer: Layer):
        layer._init_cache()
        self._update_cache(layer)

        W_cache = np.sqrt(layer.W_cache / (1 - self.b2 ** (self.iteration + 1))) + self.eps
        b_cache = np.sqrt(layer.b_cache / (1 - self.b2 ** (self.iteration + 1))) + self.eps

        return W_cache, b_cache

    def _update_cache(self, layer: Layer):
        """Same as RMSProp. b2 is Rho"""
        layer.W_cache = self.b2 * layer.W_cache + (1 - self.b2) * layer.dW**2
        layer.b_cache = self.b2 * layer.b_cache + (1 - self.b2) * layer.db**2

    def _get_momentum(self, layer: Layer):
        layer._init_momentum()
        self._update_momentum(layer)

        W_momentum = layer.W_momentum / (1 - self.b1 ** (self.iteration + 1))
        b_momentum = layer.b_momentum / (1 - self.b1 ** (self.iteration + 1))

        return W_momentum, b_momentum

    def _update_momentum(self, layer: Layer):
        """update with current gradients. b1 is momentum coefficient"""
        layer.W_momentum = self.b1 * layer.W_momentum + (1 - self.b1) * layer.dW
        layer.b_momentum = self.b1 * layer.b_momentum + (1 - self.b1) * layer.db

    def step(self, layers: List[Layer]) -> None:
        lr = self.get_adjusted_lr()
        for layer in layers:

            W_cache, b_cache = self._get_cache(layer)
            W_momentum, b_momentum = self._get_momentum(layer)

            W_update = -(lr * W_momentum) / W_cache
            b_update = -(lr * b_momentum) / b_cache

            layer.W += W_update
            layer.b += b_update

        self.iteration += 1
