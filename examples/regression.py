import numpy as np

from simplenn import Network
from simplenn.layer import Dense
from simplenn.activation import ReLu
from simplenn.activation import LinearLoss
from simplenn.metrics.loss import MeanSquaredError
from simplenn.metrics.loss import MeanAbsoluteError
from simplenn.optimizers.adam import Adam

SAMPLES = 1000

fcn = lambda x: (1 - np.cos(x)) / x

X = np.linspace(-6, 6, num=SAMPLES)
y = np.array([fcn(x) for x in X])

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


class DemoNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store all blocks
        self.l1 = Dense(1, 128, alpha=0.1)
        self.activation1 = ReLu()
        self.l2 = Dense(128, 64)
        self.activation2 = ReLu()
        self.l3 = Dense(64, 1)
        self.output = LinearLoss(loss=MeanSquaredError())

    def forward(self, x, targets):
        # forward pass
        x = self.l1(x)
        x = self.activation1(x)
        x = self.l2(x)
        x = self.activation2(x)
        x = self.l3(x)
        return self.output(x, targets)


############################################################################
optimizer = Adam(lr=0.1, decay=1e-5, b1=0.9, b2=0.999)
mse = MeanSquaredError()
mae = MeanAbsoluteError()
model = DemoNetwork(optimizer=optimizer)
model.fit(X, y, epochs=1001, batch_size=64, shuffle=True, metrics=[mse, mae])
