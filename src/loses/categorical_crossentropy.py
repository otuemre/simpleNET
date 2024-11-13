from .loss import Loss
import numpy as np

class CategoricalCrossEntropy(Loss):

    def calculate(self, y_true, y_pred):
        """
        Calculate the Categorical Cross-Entropy (CCE) loss.

        Parameters:
        ----------
        y_true : numpy array
            The true labels (one-hot encoded).
        y_pred : numpy array
            The predicted probabilities for each class.

        Returns:
        -------
        float
            The calculated CCE loss.
        """

        epsilon = 1e-15

        # Clip predictions to avoid log(0) errors
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def gradient(self, y_true, y_pred):
        """
        Calculate the gradient of the CCE loss with respect to predictions.

        Parameters:
        ----------
        y_true : numpy array
            The true labels (one-hot encoded).
        y_pred : numpy array
            The predicted probabilities for each class.

        Returns:
        -------
        numpy array
            The gradient of the CCE loss with respect to the predictions.
        """

        epsilon = 1e-15

        # Clip predictions to avoid log(0) errors
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred / y_true.shape[0]