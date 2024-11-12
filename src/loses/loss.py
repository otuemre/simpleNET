from abc import ABC, abstractmethod

class Loss(ABC):

    @abstractmethod
    def calculate(self, y_true, y_pred):
        """
        Calculate the loss value.

        Parameters:
        ----------
        y_true : numpy array
            The true labels.
        y_pred : numpy array
            The predicted labels or values.

        Returns:
        -------
        float
            The calculated loss.
        """
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred):
        """
        Calculate the gradient of the loss with respect to predictions.

        Parameters:
        ----------
        y_true : numpy array
            The true labels.
        y_pred : numpy array
            The predicted labels or values.

        Returns:
        -------
        numpy array
            The gradient of the loss with respect to the predictions.
        """
        pass

