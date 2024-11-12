from ..layers.layer import Layer
import numpy as np

class LeakyReLU(Layer):

    def __init__(self, alpha=0.01):
        """
        Initialize the Leaky ReLU activation layer.

        Parameters:
        ----------
        alpha : float, optional
            The slope for negative inputs (default is 0.01).
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input_data):
        """
        Perform the forward pass using the Leaky ReLU activation function.

        Parameters:
        ----------
        input_data : numpy array
            The input data to the layer.

        Returns:
        -------
        numpy array
            Output after applying Leaky ReLU activation, allowing a small gradient for negative values.
        """

        self.input = input_data
        # Apply Leaky ReLU function
        self.output = np.where(self.input > 0, self.input, self.alpha * self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass for the Leaky ReLU activation function.

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

        # Gradient is 1 for input > 0, and alpha for input <= 0
        input_gradient = output_gradient * np.where(self.input > 0, 1, self.alpha)
        return input_gradient