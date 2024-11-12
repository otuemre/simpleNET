from ..layers.layer import Layer
import numpy as np

class Tanh(Layer):

    def forward(self, input_data):
        """
        Perform the forward pass using the Tanh activation function.

        Parameters:
        ----------
        input_data : numpy array
            The input data to the layer.

        Returns:
        -------
        numpy array
            Output after applying Tanh activation, with values between -1 and 1.
        """

        self.input = input_data
        # Apply the Tanh function to each element in the input
        self.output = np.tanh(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass for the Tanh activation function.

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
        # Compute the derivative of Tanh (1 - tanh(x)^2)
        tanh_derivative = 1 - self.output ** 2
        # Calculate the input gradient by combining the derivative with output gradient
        input_gradient = output_gradient * tanh_derivative
        return input_gradient