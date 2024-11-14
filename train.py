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
model.train(X_train, y_train, X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, patience=10, min_delta=MIN_DELTA, early_stopping=True)

# Evaluating the model
predictions = model.predict(X_train)
print("Predictions:", predictions)
print("Rounded Predictions:", np.round(predictions))