import numpy as np
from src.model import Model
from src.layers.dense import Dense
from src.activations.tanh import Tanh
from src.activations.sigmoid import Sigmoid
from src.loses.binary_crossentropy import BinaryCrossEntropy

# Creating XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Building the model
model = Model()
model.add(Dense(input_size=2, output_size=4))  # Hidden layer with 4 neurons
model.add(Tanh())
model.add(Dense(input_size=4, output_size=1))  # Output layer with 1 neuron
model.add(Sigmoid())

# Setting loss function
model.set_loss(BinaryCrossEntropy())

# Training the model
model.train(X, y, epochs=1000, learning_rate=0.1)

# Evaluating the model
predictions = model.predict(X)
print("Predictions:", predictions)
print("Rounded Predictions:", np.round(predictions))