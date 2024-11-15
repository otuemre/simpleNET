# SimpleNET: A Simple Neural Network Framework

![Python](https://img.shields.io/badge/python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

SimpleNET is a basic neural network framework built from scratch using only Python and NumPy. Created as a learning project, it implements the essential components needed to build, train, and evaluate neural networks for tasks like binary and multi-class classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training on XOR Data](#training-on-xor-data)
  - [Example Code](#example-code)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)


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

```bash
python train.py
```

This script demonstrates a basic neural network learning the XOR function—a classic test for simple neural networks.

### Example Code

Below is a basic example of how to build and train a model using SimpleNET:

```python
import numpy as np
from src.model import Model
from src.layers.dense import Dense
from src.activations.relu import ReLU
from src.activations.sigmoid import Sigmoid
from src.loses.binary_crossentropy import BinaryCrossEntropy

# Creating XOR training data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Creating XOR validation data
X_val = np.array([[0, 1], [1, 0]])
y_val = np.array([[1], [1]])

# Define Epochs, Learning Rate, and Min Delta
EPOCHS = 10000
LEARNING_RATE = 0.1
MIN_DELTA = 1e-5

# Building the model
model = Model()
model.add(Dense(input_size=2, output_size=4))  # Hidden layer with 4 neurons
model.add(ReLU())
model.add(Dense(input_size=4, output_size=1))  # Output layer with 1 neuron
model.add(Sigmoid())

# Setting loss function
model.set_loss(BinaryCrossEntropy())

# Training the model
model.train(X_train, y_train, X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, patience=5, min_delta=MIN_DELTA, early_stopping=True)

# Evaluating the model
predictions = model.predict(X_train)
print("Predictions:", predictions)
print("Rounded Predictions:", np.round(predictions))
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

## Future Improvements

Some ideas for future improvements include:

- Adding more activation functions (e.g., Swish, GELU)
- Implementing optimizers like Adam or RMSprop
- Adding support for dropout layers for better generalization
- Expanding to convolutional layers for image data processing
- Adding batching support for efficient training on larger datasets

## Contributing

Contributions are welcome! If you would like to improve this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-YourFeatureName`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-YourFeatureName`).
5. Open a Pull Request.

Please make sure your code follows the project's coding standards and includes relevant documentation.

Thank you for your interest in contributing!

## Acknowledgments

- Inspired by various online tutorials and resources in neural network development.
- Special thanks to my friend [Tudor Hirtopanu](https://github.com/tudorhirtopanu) for inspiration from their project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

This project was created as part of a learning journey in understanding neural networks and machine learning principles.
