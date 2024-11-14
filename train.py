import numpy as np
from src.model import Model
from src.layers.dense import Dense
from src.activations.relu import ReLU
from src.activations.sigmoid import Sigmoid
from src.loses.binary_crossentropy import BinaryCrossEntropy

# Creating XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define Epochs and Learning Rate
EPOCHS = 10000
LEARNING_RATE = 0.1

# Building the model
model = Model()
model.add(Dense(input_size=2, output_size=4))  # Hidden layer with 4 neurons
model.add(ReLU())
model.add(Dense(input_size=4, output_size=1))  # Output layer with 1 neuron
model.add(Sigmoid())

# Setting loss function
model.set_loss(BinaryCrossEntropy())

# Training the model
model.train(X, y, epochs=EPOCHS, learning_rate=LEARNING_RATE)

# Evaluating the model
predictions = model.predict(X)
print("Predictions:", predictions)
print("Rounded Predictions:", np.round(predictions))