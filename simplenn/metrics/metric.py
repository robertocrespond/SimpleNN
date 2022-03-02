from abc import ABC

import numpy as np


class Metric(ABC):
    """Abstract class for evaluation metrics."""

    def __init__(self) -> None:
        self.name: str
        self.alias: str

    def __call__(self, y_hat: np.ndarray, targets: np.ndarray):
        """
        Calculate evaluation metric using predicted labels/target and ground truth

        Args:
            y_hat (np.ndarray): Predicted labels/target
            targets (np.ndarray): Ground thruth labels/targets

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
