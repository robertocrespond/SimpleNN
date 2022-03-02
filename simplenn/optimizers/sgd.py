from simplenn.layer import Layer
from simplenn.optimizers.optimizer import Optimizer
from typing import List
from typing import Union


class SGD(Optimizer):
    """
    Initialize Stochastic gradient descent optimizer

    Args:
        lr (float, optional): learning rate. Defaults to 0.1.
        decay (Union[float, None], optional): factor of linear decay per iteration to learning rate. Defaults to None.
        momentum (Union[float, None], optional): Factor of how much momentum (weight of previous gradient) to include.
            A momentum of 0 would yield regular SGD. Defaults to None.
    """

    def __init__(
        self,
        lr: float = 0.1,
        decay: Union[float, None] = None,
        momentum: Union[float, None] = None,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.decay = decay
        self.momentum = momentum

    def _get_momentum(self, layer: Layer):
        if self.momentum is None:
            return 0, 0

        layer._init_momentum()
        return self.momentum * layer.W_momentum, self.momentum * layer.b_momentum

    def _update_momentum(self, layer: Layer, W_update, b_update):
        if self.momentum is None:
            return

        layer.W_momentum = W_update
        layer.b_momentum = b_update

    def step(self, layers: List[Layer]) -> None:
        lr = self.get_adjusted_lr()
        for layer in layers:

            W_momentum, b_momentum = self._get_momentum(layer)

            W_update = W_momentum - (lr * layer.dW)
            b_update = b_momentum - (lr * layer.db)

            layer.W += W_update
            layer.b += b_update

            self._update_momentum(layer, W_update, b_update)

        self.iteration += 1
