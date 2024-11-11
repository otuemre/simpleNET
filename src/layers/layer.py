from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        """
        Initialize the layer's basic attributes.

        This constructor sets up the necessary attributes that each layer will use to manage
        the data flow and store results within the network.

        Attributes:
        ----------
        input : numpy array or None
            Stores the input data fed to this layer during the forward pass.
        output : numpy array or None
            Holds the output data produced by this layer after processing the input.

        Note:
            This base `Layer` class is designed to be extended, so the `input` and `output`
            attributes act as placeholders. Subclasses will use these attributes to track
            data as it flows through and is transformed by each specific layer type.
        """

        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_data):
        """
        Perform the forward pass through the layer.

        This method should be overridden by subclasses to implement specific layer functionality.

        :param input_data: numpy array
            The input data to the layer.
        :return: numpy array
            The output data produced by the layer after applying the layer-specific transformation.

        Note:
            This method should set the `self.input` and `self.output` attributes.
            Subclasses must implement this method to define the layer's behavior.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass for the layer.

        This method should be overridden by subclasses to calculate the gradient and update parameters.

        :param output_gradient: numpy array
            The gradient of the loss with respect to the output of this layer.
        :param learning_rate: float
            The learning rate for updating parameters, if applicable.
        :return: numpy array
            The gradient of the loss with respect to the input of this layer.

        Note:
            This method enables backpropagation by calculating the necessary parameter updates
            for this layer (if it has trainable parameters) and passes the input gradient
            for further propagation through the network.
        """
        pass