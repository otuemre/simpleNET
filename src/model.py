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

    def train(self, X, y, epochs, learning_rate, patience=5, min_delta=1e-4):
        """Train the model"""

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            # Calculate Accuracy
            accuracy = calculate_accuracy(y, y_pred)
            # Calculate loss
            loss = self.loss.calculate(y, y_pred)
            # Backward pass
            self.backward(y, y_pred, learning_rate)

            if loss < self.best_val_loss - min_delta:
                self.best_val_loss = loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}, Accuracy: {accuracy}%")

    def predict(self, X):
        """Make prediction with the trained model"""
        return self.forward(X)