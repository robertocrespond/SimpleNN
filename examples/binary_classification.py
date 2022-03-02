import numpy as np

from simplenn import Network
from simplenn.layer import Dense
from simplenn.activation import ReLu
from simplenn.activation import SigmoidLoss
from simplenn.metrics import AucPRC
from simplenn.metrics import AucROC
from simplenn.metrics.loss import BinaryCrossEntropy
from simplenn.metrics import Accuracy

from simplenn.optimizers import RMSProp

SAMPLES = 10000
FEATURES = 5
N_CLASSES = 2

X = np.random.random((SAMPLES, FEATURES))
y = np.random.binomial(N_CLASSES - 1, 1 / N_CLASSES, SAMPLES)
y_vector = y.reshape(-1, 1)  # Input to model targets must converted to column vector


class DemoNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = Dense(FEATURES, 64, W_l1=5e-4, b_l1=5e-4)
        self.activation1 = ReLu()
        self.l2 = Dense(64, 1)
        self.output = SigmoidLoss(loss=BinaryCrossEntropy())

    def forward(self, x, targets):
        # forward pass
        x = self.l1(x)
        x = self.activation1(x)
        x = self.l2(x)
        return self.output(x, targets)


optimizer = RMSProp(decay=1e-3, rho=0.999, lr=0.02)
acc = Accuracy()
model = DemoNetwork(optimizer=optimizer)

# All metrics derived from a decision function utilize a .5 threshold
model.fit(X, y_vector, epochs=10, metrics=[acc, AucROC(), AucPRC()])

yprob_train = model.predict(X)
train_acc = acc(yprob_train, y_vector)

print(f"Train Accuracy: {train_acc}")
