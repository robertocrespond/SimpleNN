# SimpleNN

SimpleNN is a simplified framework for building and training neural networks models. Framework is built on modularity and flexibility, enabling easy adoption and feature expansion.

Currently SimpleNN only supports feedforward models. Only sequential execution graphs are supported. The following operations are not currently supported:

- Concatenations of any kind (Residual gates or multiple inputs)
- Addition/Multiplications of layers
- Loops (RNNs)

### **For complete examples check the /examples directory.**

<br>

## Getting Started

### Install

```python
pip install simple-neural
```

<br>

### Creating your first model

**Example for building a multi-class deep feedforward network**

1. Load data into predictos and one hot encoded target

```python
# This generates dummy data, with 3 classes to predict
# In a real-project, load the data, pre-process it and continue to step 2
SAMPLES = 1000
FEATURES = 5
N_CLASSES = 3

X = np.random.random((SAMPLES, FEATURES))
y = np.random.binomial(N_CLASSES - 1, 1 / N_CLASSES, SAMPLES)
y_ohe = np.eye(N_CLASSES)[y]  # Input to model targets must be one hot encoded
```

2. Define network object

```python
from simplenn import Network
from simplenn.layer import Dense
from simplenn.activation import ReLu
from simplenn.activation import SoftMaxLoss
from simplenn.metrics.loss import CategoricalCrossEntropy

class DemoNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store all layers that will be part of the model

        # Dense layer, input size matches data shape,
        # output is chosen to be of size 64
        self.l1 = Dense(FEATURES, 64, W_l2=1e-3, b_l2=1e-3)

        # Rectified Linear Unit
        self.activation1 = ReLu()

        # Dense layer, input size matches previous layer output shape,
        # output is chosen to be of size of number of classes
        self.l2 = Dense(64, N_CLASSES)

        # Final activation function is chosen to be softmax
        # The final activation prediction is evaluated using
        # CategoricalCrossEntropy
        self.output = SoftMaxLoss(loss=CategoricalCrossEntropy())

    def forward(self, x, targets):
        # Define order of execution of layers for the forward pass
        x = self.l1(x)
        x = self.activation1(x)
        x = self.l2(x)
        return self.output(x, targets)
```

3. Initialize model

```python
from simplenn.optimizers import Adam

# Define optimizer
optimizer = Adam(lr=0.03, decay=5e-7, b1=0.9, b2=0.999)

# Define model
model = DemoNetwork(optimizer=optimizer)

```

4. Train model

```python
from simplenn.metrics import Accuracy

# Train model and specify metrics to track
model.fit(X, y, epochs=1000, metrics=[Accuracy()])
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

Originally built as a learning excercise.
