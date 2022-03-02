from simplenn.block import Block
from typing import Union

import numpy as np


class Sigmoid(Block):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, block: Union[Block, np.ndarray], inference: bool = False, targets=None):
        x = self.register_block(block, inference)
        output = 1 / (1 + np.exp(-x))

        if inference:
            return output
        self.output = output  # type: ignore
        return self.output  # type: ignore

    def back(self, z):
        # derivative from sigmoid function
        zstate = z * (1 - self.output) * self.output
        return zstate


class SigmoidLoss(Block):
    def __init__(self, loss) -> None:
        super().__init__()
        self.loss = loss
        self.activation = Sigmoid()

    def __call__(self, block, inference: bool = False, targets=None):
        x = self.register_block(block)
        output = self.activation(x)
        if inference:
            return output
        self.output = output
        return self.output, self.loss(self.output, targets)

    def back(self, z, targets):
        z = self.loss.back(z, targets)
        zstate = self.activation.back(z)
        return zstate
