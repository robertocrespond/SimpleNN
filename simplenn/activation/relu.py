from simplenn.block import Block
from typing import Union

import numpy as np


class ReLu(Block):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, block: Union[Block, np.ndarray], inference: bool = False, targets=None):
        x = self.register_block(block, inference)
        output = np.maximum(0, x)
        if inference:
            return output
        self.x = x
        self.output = output  # type: ignore
        return self

    def back(self, z):
        zstate = z.copy()
        zstate[self.x <= 0] = 0
        return zstate
