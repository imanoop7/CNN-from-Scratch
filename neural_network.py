import numpy as np
from activation_functions import sigmoid, sigmoid_derivative, softmax
from loss_functions import cross_entropy_loss, cross_entropy_loss_derivative

class ConvLayer:
    def __init__(self, num_filters, filter_size, input_depth):
        print(f"Initializing ConvLayer with {num_filters} filters, filter size {filter_size}, input depth {input_depth}")
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_depth = input_depth
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)  # Changed shape to (num_filters,)
        print(f"ConvLayer initialized with filters shape {self.filters.shape} and biases shape {self.biases.shape}")
    
    def forward(self, input):
        print("ConvLayer forward pass started.")
        self.input = input
        batch_size, input_depth, input_height, input_width = input.shape
        filter_height, filter_width = self.filter_size, self.filter_size
        output_height = input_height - filter_height + 1
        output_width = input_width - filter_width + 1

        print(f"Input shape: {input.shape}")
        print(f"Output shape after convolution: ({batch_size}, {self.num_filters}, {output_height}, {output_width})")

        # Initialize output
        self.output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        
        for i in range(output_height):
            for j in range(output_width):
                input_slice = input[:, :, i:i+filter_height, j:j+filter_width]  # Shape: (batch_size, input_depth, filter_height, filter_width)
                # Perform convolution using tensordot for efficiency
                # The result shape will be (batch_size, num_filters)
                self.output[:, :, i, j] = np.tensordot(input_slice, self.filters, axes=([1,2,3], [1,2,3])) + self.biases
                if i == 0 and j == 0:
                    print(f"ConvLayer forward slice ({i}, {j}): input_slice shape {input_slice.shape}, output[:, :, i, j] shape {self.output[:, :, i, j].shape}")
        print("ConvLayer forward pass completed.")
        return self.output

    def backward(self, d_out, learning_rate):
        print("ConvLayer backward pass started.")
        batch_size, _, output_height, output_width = d_out.shape
        filter_height, filter_width = self.filter_size, self.filter_size

        # Initialize gradients
        d_filters = np.zeros_like(self.filters)
        d_biases = np.sum(d_out, axis=(0, 2, 3))  # Shape: (num_filters,)
        d_input = np.zeros_like(self.input)

        print(f"d_out shape: {d_out.shape}")
        print(f"d_filters shape: {d_filters.shape}")
        print(f"d_biases shape: {d_biases.shape}")
        print(f"d_input shape: {d_input.shape}")

        for i in range(output_height):
            for j in range(output_width):
                input_slice = self.input[:, :, i:i+filter_height, j:j+filter_width]  # Shape: (batch_size, input_depth, filter_height, filter_width)
                # Gradient w.r.t. filters
                for f in range(self.num_filters):
                    gradient = np.sum(input_slice * d_out[:, f, i, j][:, None, None, None], axis=0)
                    d_filters[f] += gradient
                    if i == 0 and j == 0 and f == 0:
                        print(f"Gradient for filter {f} at slice ({i}, {j}): {gradient}")
                
                # Gradient w.r.t. input
                for f in range(self.num_filters):
                    d_input[:, :, i:i+filter_height, j:j+filter_width] += self.filters[f] * d_out[:, f, i, j][:, None, None, None]
        
        # Update filters and biases
        print("Updating filters and biases.")
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases

        print("ConvLayer backward pass completed.")
        return d_input

class MaxPoolLayer:
    def __init__(self, size=2, stride=2):
        print(f"Initializing MaxPoolLayer with size {size} and stride {stride}")
        self.size = size
        self.stride = stride
        self.max_indexes = {}
    
    def forward(self, input):
        print("MaxPoolLayer forward pass started.")
        self.input = input
        batch_size, depth, input_height, input_width = input.shape
        out_height = (input_height - self.size) // self.stride + 1
        out_width = (input_width - self.size) // self.stride + 1

        print(f"Input shape: {input.shape}")
        print(f"Output shape after pooling: ({batch_size}, {depth}, {out_height}, {out_width})")

        self.output = np.zeros((batch_size, depth, out_height, out_width))
        
        for b in range(batch_size):
            for d_layer in range(depth):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = input[b, d_layer, h_start:h_start+self.size, w_start:w_start+self.size]
                        max_val = np.max(window)
                        self.output[b, d_layer, i, j] = max_val
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indexes[(b, d_layer, i, j)] = (h_start + max_pos[0], w_start + max_pos[1])
                        if b == 0 and d_layer == 0 and i == 0 and j == 0:
                            print(f"MaxPoolLayer forward window ({i}, {j}) in depth {d_layer}, batch {b}: max_val={max_val} at position {self.max_indexes[(b, d_layer, i, j)]}")
        print("MaxPoolLayer forward pass completed.")
        return self.output

    def backward(self, d_out):
        print("MaxPoolLayer backward pass started.")
        batch_size, depth, out_height, out_width = d_out.shape
        d_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for d_layer in range(depth):
                for i in range(out_height):
                    for j in range(out_width):
                        h, w = self.max_indexes[(b, d_layer, i, j)]
                        d_input[b, d_layer, h, w] += d_out[b, d_layer, i, j]
                        if b == 0 and d_layer == 0 and i == 0 and j == 0:
                            print(f"MaxPoolLayer backward window ({i}, {j}) in depth {d_layer}, batch {b}: d_out={d_out[b, d_layer, i, j]}, updated d_input[{b}, {d_layer}, {h}, {w}]={d_input[b, d_layer, h, w]}")
        print("MaxPoolLayer backward pass completed.")
        return d_input

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        print(f"Initializing FullyConnectedLayer with input size {input_size} and output size {output_size}")
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        print(f"FullyConnectedLayer initialized with weights shape {self.weights.shape} and bias shape {self.bias.shape}")
    
    def forward(self, input):
        print("FullyConnectedLayer forward pass started.")
        self.input = input  # Shape: (batch_size, input_size)
        self.output = np.dot(self.input, self.weights) + self.bias  # Shape: (batch_size, output_size)
        print(f"FullyConnectedLayer forward output shape: {self.output.shape}")
        return self.output
    
    def backward(self, d_out, learning_rate):
        print("FullyConnectedLayer backward pass started.")
        d_weights = np.dot(self.input.T, d_out)  # Shape: (input_size, output_size)
        d_bias = np.sum(d_out, axis=0, keepdims=True)  # Shape: (1, output_size)
        d_input = np.dot(d_out, self.weights.T)  # Shape: (batch_size, input_size)
        
        print(f"Gradient shapes - d_weights: {d_weights.shape}, d_bias: {d_bias.shape}, d_input: {d_input.shape}")

        # Update weights and biases
        print("Updating weights and biases in FullyConnectedLayer.")
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        
        print("FullyConnectedLayer backward pass completed.")
        return d_input

class NeuralNetwork:
    def __init__(self):
        print("Initializing NeuralNetwork.")
        # CNN Architecture:
        # 1. Convolutional Layer
        # 2. Pooling Layer
        # 3. Fully Connected Layer
        self.conv = ConvLayer(num_filters=6, filter_size=5, input_depth=1)
        self.pool = MaxPoolLayer(size=2, stride=2)
        self.fc_input_size = 6 * 12 * 12  # Adjust based on input dimensions (assuming input 28x28)
        self.fc = FullyConnectedLayer(self.fc_input_size, 10)
        print("NeuralNetwork initialization completed.")
    
    def forward(self, X):
        print("NeuralNetwork forward pass started.")
        self.conv_output = self.conv.forward(X)  # Shape: (batch_size, 6, 24, 24)
        print(f"After ConvLayer: {self.conv_output.shape}")
        
        self.pool_output = self.pool.forward(self.conv_output)  # Shape: (batch_size, 6, 12, 12)
        print(f"After MaxPoolLayer: {self.pool_output.shape}")
        
        self.flatten = self.pool_output.reshape(X.shape[0], -1)  # Shape: (batch_size, 864)
        print(f"After flattening: {self.flatten.shape}")
        
        self.fc_output = softmax(self.fc.forward(self.flatten))  # Shape: (batch_size, 10)
        print(f"After FullyConnectedLayer and softmax: {self.fc_output.shape}")
        print("NeuralNetwork forward pass completed.")
        return self.fc_output
    
    def backward(self, X, y, output, learning_rate):
        print("NeuralNetwork backward pass started.")
        # Compute loss derivative
        loss_derivative = cross_entropy_loss_derivative(y, output)  # Shape: (batch_size, 10)
        print(f"Loss derivative shape: {loss_derivative.shape}")
        
        # Backward through Fully Connected Layer
        d_fc = self.fc.backward(loss_derivative, learning_rate)  # Shape: (batch_size, 864)
        print(f"Gradient after FullyConnectedLayer: {d_fc.shape}")
        
        # Backward through Flatten
        d_pool = d_fc.reshape(self.pool_output.shape)  # Shape: (batch_size, 6, 12, 12)
        print(f"Gradient after reshaping for Pooling Layer: {d_pool.shape}")
        
        # Backward through Pooling Layer
        d_conv = self.pool.backward(d_pool)  # Shape: (batch_size, 6, 24, 24)
        print(f"Gradient after MaxPoolLayer backward: {d_conv.shape}")
        
        # Backward through Convolutional Layer
        d_input = self.conv.backward(d_conv, learning_rate)  # Shape: (batch_size, 1, 28, 28)
        print(f"Gradient after ConvLayer backward: {d_input.shape}")
        
        print("NeuralNetwork backward pass completed.")
        return d_input
    
    def train(self, X, y, epochs, learning_rate, batch_size):
        print("Starting training process.")
        n_samples = X.shape[0]
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} started.")
            # Shuffle the data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            print("Data shuffled.")
            
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                print(f"Processing batch {i//batch_size + 1} with batch size {X_batch.shape[0]}")
                output = self.forward(X_batch)
                loss = cross_entropy_loss(y_batch, output)
                print(f"Batch loss: {loss:.4f}")
                total_loss += loss
                self.backward(X_batch, y_batch, output, learning_rate)
            
            avg_loss = total_loss / (n_samples / batch_size)
            print(f'Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}\n')
        print("Training completed.")