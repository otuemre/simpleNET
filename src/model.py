from .util.accuracy import calculate_accuracy

class Model:

    def __init__(self):
        self.layers = []
        self.loss = None

        # Early Stopping
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

    def add(self, layer):
        """Add a layer to the model"""
        self.layers.append(layer)

    def set_loss(self, loss):
        """Set the loss function for the model"""
        self.loss = loss

    def forward(self, X):
        """Perform the forward pass"""
        # Pass input through each layer
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred, learning_rate):
        """Perform the backward pass"""
        # Calculate initial gradient from the loss
        loss_gradient = self.loss.gradient(y_true, y_pred)
        # Back propagate through each layer in reverse
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, patience=5, min_delta=1e-4, early_stopping=False):
        """Train the model with optional early stopping"""

        for epoch in range(epochs):
            # Forward pass on training data
            y_pred_train = self.forward(X_train)
            train_accuracy = calculate_accuracy(y_train, y_pred_train)
            train_loss = self.loss.calculate(y_train, y_pred_train)
            self.backward(y_train, y_pred_train, learning_rate)

            # Validation loss and early stopping check if enabled
            if early_stopping:
                y_pred_val = self.forward(X_val)
                val_loss = self.loss.calculate(y_val, y_pred_val)

                # Use the helper function for early stopping logic
                if self._check_early_stopping(val_loss, min_delta, patience):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%, Val Loss: {val_loss}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%")

    def _check_early_stopping(self, val_loss, min_delta, patience):
        """Check and update early stopping condition"""

        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        # Return True if early stopping should trigger
        return self.early_stopping_counter >= patience

    def predict(self, X):
        """Make prediction with the trained model"""
        return self.forward(X)