import numpy as np

from simplenn import Network
from simplenn.layer import Dense
from simplenn.activation import ReLu
from simplenn.activation import SoftMaxLoss
from simplenn.layer.dropout import Dropout
from simplenn.metrics.loss import CategoricalCrossEntropy
from simplenn.metrics import Accuracy

from simplenn.optimizers import Adam

SAMPLES = 1000
FEATURES = 5
N_CLASSES = 3

X = np.random.random((SAMPLES, FEATURES))
y = np.random.binomial(N_CLASSES - 1, 1 / N_CLASSES, SAMPLES)
y_ohe = np.eye(N_CLASSES)[y]  # Input to model targets must be one hot encoded


class DemoNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = Dense(FEATURES, 64, W_l1=5e-4, b_l1=5e-4)
        self.dropout1 = Dropout(rate=0.2)
        self.activation1 = ReLu()
        self.l2 = Dense(64, 32)
        self.dropout2 = Dropout(rate=0.2)
        self.activation2 = ReLu()
        self.l3 = Dense(32, N_CLASSES)
        self.output = SoftMaxLoss(loss=CategoricalCrossEntropy())

    def forward(self, x, targets):
        # forward pass
        x = self.l1(x)
        x = self.dropout1(x)
        x = self.activation1(x)
        x = self.l2(x)
        x = self.dropout2(x)
        x = self.activation2(x)
        x = self.l3(x)
        return self.output(x, targets)


optimizer = Adam(lr=0.03, decay=5e-4, b1=0.9, b2=0.999)
acc = Accuracy()
model = DemoNetwork(optimizer=optimizer)

model.fit(X, y_ohe, epochs=100, batch_size=512, metrics=[acc])

yprob_train = model.predict(X)
train_acc = acc(yprob_train, y_ohe)

print(f"Train Accuracy: {train_acc}")
