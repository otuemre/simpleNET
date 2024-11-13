from .loss import Loss
import numpy as np

class BinaryCrossEntropy(Loss):

    def calculate(self, y_true, y_pred):
        """
        Calculate the Binary Cross-Entropy (BCE) loss.

        Parameters:
        ----------
        y_true : numpy array
            The true binary labels.
        y_pred : numpy array
            The predicted probabilities.

        Returns:
        -------
        float
            The calculated BCE loss.
        """

        epsilon = 1e-15

        # Clip prediction to avoid log(0) errors
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        """
        Calculate the gradient of the BCE loss with respect to predictions.

        Parameters:
        ----------
        y_true : numpy array
            The true binary labels.
        y_pred : numpy array
            The predicted probabilities.

        Returns:
        -------
        numpy array
            The gradient of the BCE loss with respect to the predictions.
        """

        epsilon = 1e-15

        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.size