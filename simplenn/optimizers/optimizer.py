from abc import ABC
from simplenn.layer.layer import Layer
from typing import List


class Optimizer(ABC):
    """Base class for optimizers of block parameters"""

    def __init__(self) -> None:
        self.iteration = 0

    def get_adjusted_lr(self):
        """Learning rate adjusted by linear decay"""
        lr = self.lr
        if self.decay:
            lr = self.lr / (1 + self.decay * self.iteration)
        return lr

    def step(self, layers: List[Layer]) -> None:
        """
        Take a gradient descent step. Update parameters by one iteration.

        Args:
            layers (List[Layer]): Layers to update parameters to

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
