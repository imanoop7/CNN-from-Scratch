import numpy as np
from activation_functions import sigmoid, sigmoid_derivative, softmax
from loss_functions import cross_entropy_loss, cross_entropy_loss_derivative

class ConvLayer:
    def __init__(self, num_filters, filter_size, input_depth):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_depth = input_depth
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.1
        self.biases = np.zeros((num_filters, 1))
    
    def forward(self, input):
        self.input = input
        batch_size, input_depth, input_height, input_width = input.shape
        filter_height, filter_width = self.filter_size, self.filter_size
        output_height = input_height - filter_height + 1
        output_width = input_width - filter_width + 1
        self.output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        region = input[b, :, i:i+filter_height, j:j+filter_width]
                        self.output[b, f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        return self.output
    
    def backward(self, d_out, learning_rate):
        batch_size, _, _, _ = self.input.shape
        _, _, output_height, output_width = d_out.shape
        filter_height, filter_width = self.filter_size, self.filter_size

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        region = self.input[b, :, i:i+filter_height, j:j+filter_width]
                        d_filters[f] += d_out[b, f, i, j] * region
                        d_biases[f] += d_out[b, f, i, j]
                        d_input[b, :, i:i+filter_height, j:j+filter_width] += d_out[b, f, i, j] * self.filters[f]
        
        # Update filters and biases
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases

        return d_input

class MaxPoolLayer:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride
    
    def forward(self, input):
        self.input = input
        batch_size, depth, height, width = input.shape
        out_height = (height - self.size) // self.stride + 1
        out_width = (width - self.size) // self.stride + 1
        self.output = np.zeros((batch_size, depth, out_height, out_width))
        self.max_indexes = {}

        for b in range(batch_size):
            for d in range(depth):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = input[b, d, h_start:h_start+self.size, w_start:w_start+self.size]
                        max_val = np.max(window)
                        self.output[b, d, i, j] = max_val
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indexes[(b, d, i, j)] = (h_start + max_pos[0], w_start + max_pos[1])
        return self.output
    
    def backward(self, d_out):
        batch_size, depth, out_height, out_width = d_out.shape
        d_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for d in range(depth):
                for i in range(out_height):
                    for j in range(out_width):
                        h, w = self.max_indexes[(b, d, i, j)]
                        d_input[b, d, h, w] += d_out[b, d, i, j]
        return d_input

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
    
    def forward(self, input):
        self.input = input  # Shape: (batch_size, input_size)
        self.output = np.dot(self.input, self.weights) + self.bias  # Shape: (batch_size, output_size)
        return self.output
    
    def backward(self, d_out, learning_rate):
        d_weights = np.dot(self.input.T, d_out)  # Shape: (input_size, output_size)
        d_bias = np.sum(d_out, axis=0, keepdims=True)  # Shape: (1, output_size)
        d_input = np.dot(d_out, self.weights.T)  # Shape: (batch_size, input_size)
        
        # Update weights and biases
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        
        return d_input

class NeuralNetwork:
    def __init__(self):
        # CNN Architecture:
        # 1. Convolutional Layer
        # 2. Pooling Layer
        # 3. Fully Connected Layer
        self.conv = ConvLayer(num_filters=6, filter_size=5, input_depth=1)
        self.pool = MaxPoolLayer(size=2, stride=2)
        self.fc_input_size = 6 * 12 * 12  # Adjust based on input dimensions (assuming input 28x28)
        self.fc = FullyConnectedLayer(self.fc_input_size, 10)
    
    def forward(self, X):
        self.conv_output = self.conv.forward(X)  # Shape: (batch_size, 6, 24, 24)
        self.pool_output = self.pool.forward(self.conv_output)  # Shape: (batch_size, 6, 12, 12)
        self.flatten = self.pool_output.reshape(X.shape[0], -1)  # Shape: (batch_size, 864)
        self.fc_output = softmax(self.fc.forward(self.flatten))  # Shape: (batch_size, 10)
        return self.fc_output
    
    def backward(self, X, y, output, learning_rate):
        # Compute loss derivative
        loss_derivative = cross_entropy_loss_derivative(y, output)  # Shape: (batch_size, 10)
        
        # Backward through Fully Connected Layer
        d_fc = self.fc.backward(loss_derivative, learning_rate)  # Shape: (batch_size, 864)
        
        # Backward through Flatten
        d_pool = d_fc.reshape(self.pool_output.shape)  # Shape: (batch_size, 6, 12, 12)
        
        # Backward through Pooling Layer
        d_conv = self.pool.backward(d_pool)  # Shape: (batch_size, 6, 24, 24)
        
        # Backward through Convolutional Layer
        d_input = self.conv.backward(d_conv, learning_rate)  # Shape: (batch_size, 1, 28, 28)
        
        return d_input
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = cross_entropy_loss(y, output)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')