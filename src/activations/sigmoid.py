from ..layers.layer import Layer
import numpy as np

class Sigmoid(Layer):

    def forward(self, input_data):
        """
        Perform the forward pass using the Sigmoid activation function.

        Parameters:
        ----------
        input_data : numpy array
            The input data to the layer.

        Returns:
        -------
        numpy array
            Output after applying Sigmoid activation, with values between 0 and 1.
        """

        self.input = input_data
        # Apply the Sigmoid function to each element in the input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass for the Sigmoid activation function.

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
        # Compute the derivative of Sigmoid
        sigmoid_derivative = self.output * (1 - self.output)
        # Calculate the input gradient by combining the derivative with output gradient
        input_gradient = output_gradient * sigmoid_derivative
        return input_gradient