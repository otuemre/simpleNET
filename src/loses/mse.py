from loss import Loss
import numpy as np

class MSE(Loss):

    def calculate(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE) loss.

        Parameters:
        ----------
        y_true : numpy array
            The true values.
        y_pred : numpy array
            The predicted values.

        Returns:
        -------
        float
            The calculated MSE loss.
        """

        return np.mean((y_pred - y_true) ** 2)

    def gradient(self, y_true, y_pred):
        """
        Calculate the gradient of the MSE loss with respect to predictions.

        Parameters:
        ----------
        y_true : numpy array
            The true values.
        y_pred : numpy array
            The predicted values.

        Returns:
        -------
        numpy array
            The gradient of the MSE loss with respect to the predictions.
        """

        return (2 / y_true.size) * (y_pred - y_true)