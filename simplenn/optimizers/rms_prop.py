from simplenn.layer import Layer
from simplenn.optimizers.optimizer import Optimizer
from typing import List
from typing import Union

import numpy as np


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation -> Evolution of AdaGrad without monotonic
        decrease of the learning rate. Includes a momentum factor (rho) of
        the partial derivates sum

    Args:
        lr (float, optional): learning rate. Defaults to 0.1.
        decay (Union[float, None], optional): factor of linear decay per iteration to learning rate. Defaults to None.
        eps (float, optional): Epsilon. Avoids division by zero. Defaults to 1e-7 .
        rho (Union[float, None], optional): Factor for decaying moving average of partial
            dertivates how much momentum (weight of previous gradient) to include.
            A rho of 0 woudl yield regular AdaGrad. Defaults to 0.9 .
    """

    def __init__(self, lr: float = 0.1, decay: Union[float, None] = None, eps: float = 1e-7, rho: float = 0.9) -> None:
        super().__init__()
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.rho = rho

    def _get_cache(self, layer: Layer):
        layer._init_cache()
        self._update_cache(layer)

        W_cache = np.sqrt(layer.W_cache) + self.eps
        b_cache = np.sqrt(layer.b_cache) + self.eps

        return W_cache, b_cache

    def _update_cache(self, layer: Layer):
        layer.W_cache = self.rho * layer.W_cache + (1 - self.rho) * layer.dW**2
        layer.b_cache = self.rho * layer.b_cache + (1 - self.rho) * layer.db**2

    def step(self, layers: List[Layer]) -> None:
        lr = self.get_adjusted_lr()
        for layer in layers:

            W_cache, b_cache = self._get_cache(layer)

            W_update = -(lr * layer.dW) / W_cache
            b_update = -(lr * layer.db) / b_cache

            layer.W += W_update
            layer.b += b_update

        self.iteration += 1
