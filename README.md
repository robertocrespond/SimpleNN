# SimpleNN

SimpleNN is a simplified framework for building and training neural networks models. Framework is build on modularity and flexibility to enable easy adoption and feature expansions.

Currently SimpleNN only supports sequential models. Only sequential execution graphs are supported. This means that the following operations are not supported:

- Concatenations of any kind (Residual gates or multiple inputs)
- Addition/Multiplications of layers
- Loops (RNNs)

Originally built as a learning excercise.

### **For complete examples check the /examples directory.**

<br>

## Getting Started

### Install

```python
pip install simple-neural
```

```python
# Multi class classification
FEATURES = 5
N_CLASSES = 3

class DemoNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = Dense(FEATURES, 64, W_l2=1e-3, b_l2=1e-3)
        self.activation1 = ReLu()
        self.l2 = Dense(32, N_CLASSES)
        self.output = SoftMaxLoss(loss=CategoricalCrossEntropy())

    def forward(self, x, targets): # define forward pass
        x = self.l1(x)
        x = self.activation1(x)
        x = self.l2(x)
        return self.output(x, targets)

optimizer = Adam(lr=0.03, decay=5e-7, b1=0.9, b2=0.999)
acc = Accuracy()
model = DemoNetwork(optimizer=optimizer)
model.fit(X, y, epochs=1000, metrics=[acc])
```

## Saving/Loading Network

```python
from simplenn import Network

# serialize
model.save('simplemodel.pkl')

# deserialize
m2 = Network.load('simplemodel.pkl')
yhat = m2.predict(X)
```

## Features

### Layers

- Dense : simplenn.layers.Dense
- Dropout : simplenn.layers.Dropout

### Activations

- ReLu : simplenn.activation.ReLu
- Sigmoid : simplenn.activation.Sigmoid
- SoftMax : simplenn.activation.SoftMax
- Linear : simplenn.activation.Linear

### Losses

- Binary cross entropy : simplenn.metrics.loss.BinaryCrossEntropy
- Categorical cross entropy : simplenn.metrics.loss.CategoricalCrossEntropy
- Mean absolute error : simplenn.metrics.loss.MeanAbsoluteError
- Mean squared error : simplenn.metrics.loss.MeanSquaredError

### Optimizers

- Stochastic Gradient Descent : simplenn.optimizers.SGD
- Adaptive Gradient : simplenn.optimizers.AdaGrad
- Root Mean Squared Propagation : simplenn.optimizers.RMSProp
- Adaptive Moment Estimation : simplenn.optimizers.Adam

### Metrics

- Accuracy : simplenn.metrics.Accuracy
- Mean absolute error : simplenn.metrics.loss.MeanAbsoluteError
- Mean squared error : simplenn.metrics.loss.MeanSquaredError
- Area under the curve - receiver operating characteristic : simplenn.metrics.AucROC
- Area under the curve - precision recall curve : simplenn.metrics.AucPRC
- TPR (Recall) : simplenn.metrics.TPR
- FPR : simplenn.metrics.FPR
