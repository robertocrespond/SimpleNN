from simplenn.metrics.loss.loss import Loss

import numpy as np


class MeanSquaredError(Loss):
    name = "mean_squared_error"
    alias = "mse"

    def __call__(self, prediction, targets):
        loss_vector = np.mean((targets - prediction) ** 2, axis=-1)
        return np.mean(loss_vector)

    def back(self, z, targets):
        zstate = -2 * (targets - z) / len(z[0])
        zstate /= len(z)
        return zstate
