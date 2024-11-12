from ..layers.layer import Layer
import numpy as np

class ReLU(Layer):

    def forward(self, input_data):
        """
        Perform the forward pass using the ReLU activation function.

        Parameters:
        ----------
        input_data : numpy array
            The input data to the layer.

        Returns:
        -------
        numpy array
            Output after applying ReLU activation, where negative values are set to zero.
        """

        # Initializing input
        self.input = input_data

        # Normalizing output
        self.output = np.maximum(0, self.input)

        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass for the ReLU activation function.

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

        # Calculate and return the gradient with respect to ReLU
        input_gradient = output_gradient * (self.input > 0)
        return input_gradient