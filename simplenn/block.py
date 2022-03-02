from abc import ABC
from typing import Union

import numpy as np


class Block(ABC):
    """Base building block for neural network components"""

    def __init__(self) -> None:
        self.output = None
        self.x: np.ndarray
        self.dW: np.ndarray
        self.db: np.ndarray
        self.prev_block: Union[Block, None] = None
        self.next_block: Union[Block, None] = None

    def __call__(self, block, inference: bool = False, targets=None):
        """Forward pass operation of block"""
        raise NotImplementedError

    def back(self, block, inference=False):
        """Backward pass operation of block"""
        raise NotImplementedError

    def register_block(self, block: Union["Block", np.ndarray], inference: bool = False) -> np.ndarray:
        """
        Registers neighboring blocks in the defined execution for the forward pass

        Args:
            block (Block): Previous block in defined execution for the forward pass.
            inference (bool, optional): If enabled, state is not recorded. Defaults to False.

        Returns:
            np.ndarray: Output of previous block.
        """
        x = block
        if isinstance(block, Block) and not inference:
            block.next_block = self
            self.prev_block = block
            x = x.output  # type: ignore
        return x  # type: ignore

    def cleanup(self) -> None:
        """Delete state used for fitting model"""
        if hasattr(self, "output"):
            del self.output
        if hasattr(self, "x"):
            del self.x
        if hasattr(self, "dW"):
            del self.dW
        if hasattr(self, "db"):
            del self.db
