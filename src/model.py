class Model:

    def __init__(self):
        self.layers = []
        self.loss = None

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

    def train(self, X, y, epochs, learning_rate):
        """Train the model"""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            # Calculate loss
            loss = self.loss.calculate(y, y_pred)
            # Backward pass
            self.backward(y, y_pred, learning_rate)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def predict(self, X):
        """Make prediction with the trained model"""
        return self.forward(X)