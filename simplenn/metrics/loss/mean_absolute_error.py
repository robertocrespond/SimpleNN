from simplenn.metrics.loss.loss import Loss

import numpy as np


class MeanAbsoluteError(Loss):
    name = "mean_absolute_error"
    alias = "mae"

    def __call__(self, prediction, targets):
        loss_vector = np.mean(np.abs(targets - prediction), axis=-1)
        return np.mean(loss_vector)

    def back(self, z, targets):
        zstate = np.sign(targets - z) / len(z[0])
        zstate /= len(z)
        return zstate
