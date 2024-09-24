import numpy as np
from activation_functions import sigmoid, sigmoid_derivative, softmax
from loss_functions import cross_entropy_loss, cross_entropy_loss_derivative

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize biases
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
    
    def forward(self, X):
        # Flatten the input
        X_flat = X.reshape(X.shape[0], -1)
        
        # Forward pass
        self.hidden_input = np.dot(X_flat, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_output = softmax(self.output_input)
        return self.output_output
    
    def backward(self, X, y, output):
        # Flatten the input
        X_flat = X.reshape(X.shape[0], -1)
        
        # Backward pass
        output_error = cross_entropy_loss_derivative(y, output)
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output -= np.dot(self.hidden_output.T, output_error)
        self.bias_output -= np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= np.dot(X_flat.T, hidden_error)
        self.bias_hidden -= np.sum(hidden_error, axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = cross_entropy_loss(y, output)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
