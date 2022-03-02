from simplenn.layer import Layer
from simplenn.metrics.metric import Metric

import numpy as np


class Loss(Metric):
    """Base class for evaluation metrics that can be used as losses during backpropagation."""

    def get_reg_loss(self, layer: Layer):
        """Calculates loss attributable to regulation L1 and L2 factors."""
        attrs = ["W", "b", "W_l1", "b_l1", "W_l2", "b_l2"]
        reg_loss = 0

        if not all([hasattr(layer, at) for at in attrs]):
            return reg_loss

        if layer.W_l1 > 0:
            reg_loss += layer.W_l1 * np.sum(np.abs(layer.W))
        if layer.b_l1 > 0:
            reg_loss += layer.b_l1 * np.sum(np.abs(layer.b))
        if layer.W_l2 > 0:
            reg_loss += layer.W_l2 * np.sum(layer.W**2)
        if layer.b_l2 > 0:
            reg_loss += layer.b_l2 * np.sum(layer.b**2)

        return reg_loss
