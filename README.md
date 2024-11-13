# SimpleNET: A Simple Neural Network Framework

SimpleNET is a basic neural network framework built from scratch using only Python and NumPy. Created as a learning project, it implements the essential components needed to build, train, and evaluate neural networks for tasks like binary and multi-class classification.

## Project Overview

This project demonstrates a foundational understanding of neural networks by building a simple framework without high-level libraries. By focusing on the core elements, SimpleNET serves as a great learning tool for anyone looking to understand the inner workings of neural networks.

## Features

- **Layers**: Custom dense layers with flexible activation functions
- **Activation Functions**: Includes ReLU, Sigmoid, Tanh, Leaky ReLU, and Softmax
- **Loss Functions**: Implements Mean Squared Error, Binary Cross-Entropy, and Categorical Cross-Entropy
- **Training**: Supports backpropagation and gradient descent for learning
- **Flexible Model Class**: Add layers, specify loss functions, and train or predict with ease

## Getting Started

To get started with SimpleNET, clone the repository and install the required dependencies.

### Prerequisites

- Python 3.x
- NumPy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/otuemre/simpleNET
   ```
2. Navigate to the project directory:
   ```bash
   cd simplenet
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training on XOR Data

To see SimpleNET in action, you can train it on the XOR problem using the provided `train.py`:

```python
python train.py
```

This script demonstrates a basic neural network learning the XOR function—a classic test for simple neural networks.

### Example Code

Below is a basic example of how to build and train a model using SimpleNET:

```python
from src.model import Model
from src.layers.dense import Dense
from src.activations.relu import ReLU
from src.activations.sigmoid import Sigmoid
from src.loses.binary_crossentropy import BinaryCrossEntropy
import numpy as np

# Sample XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Build the model
model = Model()
model.add(Dense(input_size=2, output_size=4))
model.add(ReLU())
model.add(Dense(input_size=4, output_size=1))
model.add(Sigmoid())
model.set_loss(BinaryCrossEntropy())

# Train the model
model.train(X, y, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)
```

## Project Structure

```
simplenet/
├── src/
│   ├── layers/            # Contains the dense layer implementations
│   ├── activations/       # Different activation functions like ReLU, Sigmoid, Tanh
│   ├── losses/            # Loss functions for binary and multi-class tasks
│   ├── model.py           # Main model class
├── train.py               # Script to train and test the model
└── requirements.txt       # Project dependencies
```

---

This project was created as part of a learning journey in understanding neural networks and machine learning principles.
