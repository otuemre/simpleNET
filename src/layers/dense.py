from .layer import Layer
import numpy as np

class Dense(Layer):

    def __init__(self, input_size, output_size):
        """
        Initialize the dense layer with weights and biases.

        Parameters:
        ----------
        input_size : int
            The number of input features to this layer.
        output_size : int
            The number of neurons (outputs) in this layer.

        Attributes:
        ----------
        weights : numpy array
            The weight matrix, initialized with small random values.
        biases : numpy array
            The bias vector, initialized to zeros.
        """
        super().__init__()

        # Initializing Weights and Biases
        self.weights = np.random.rand(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        """
        Perform the forward pass through the dense layer.

        Parameters:
        ----------
        input_data : numpy array
            The input data to the layer with shape (batch_size, input_size).

        Returns:
        -------
        numpy array
            The output of the layer after applying the weight and bias transformation,
            with shape (batch_size, output_size).
        """

        self.input = input_data

        # Calculating output
        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass for the dense layer.

        Parameters:
        ----------
        output_gradient : numpy array
            Gradient of the loss with respect to the output of this layer.
        learning_rate : float
            Learning rate to scale the parameter updates.

        Returns:
        -------
        numpy array
            Gradient of the loss with respect to the input of this layer.
        """

        # Calculate gradients for weights and biases
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        # Calculate and return the gradient with respect to the input
        input_gradient = np.dot(output_gradient, self.weights.T)
        return  input_gradient
