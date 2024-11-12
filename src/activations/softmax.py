from ..layers.layer import Layer
import numpy as np

class Softmax(Layer):

    def forward(self, input_data):
        """
        Perform the forward pass using the Softmax activation function.

        Parameters:
        ----------
        input_data : numpy array
            The input data to the layer.

        Returns:
        -------
        numpy array
            Output after applying Softmax activation, representing class probabilities.
        """

        self.input = input_data
        # Subtract max for numerical stability
        exp_values = np.exp(self.input - np.max(self.input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass for the Softmax activation function.

        Parameters:
        ----------
        output_gradient : numpy array
            The gradient of the loss with respect to the output of this layer.
        learning_rate : float
            Not used in activation layers, but kept for consistency.

        Returns:
        -------
        numpy array
            Gradient of the loss with respect to the input of this layer.
        """

        # For Softmax with cross-entropy loss, the gradient is simply output - target
        input_gradient = output_gradient * (self.output * (1 - self.output))
        return input_gradient