import numpy as np
from neural_network import NeuralNetwork
from data_loader import load_data

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Convert labels to one-hot encoding
    y_train_one_hot = np.zeros((y_train.size, 10))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1
    
    # Network parameters
    input_size = X_train.shape[1]  # Flattened image size (28*28)
    hidden_size = 128
    output_size = 10  # Number of classes in MNIST
    epochs = 1000
    learning_rate = 0.1
    
    # Initialize and train the network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, y_train_one_hot, epochs, learning_rate)
    
    # Save the trained weights and biases
    np.save('weights_input_hidden.npy', nn.weights_input_hidden)
    np.save('weights_hidden_output.npy', nn.weights_hidden_output)
    np.save('bias_hidden.npy', nn.bias_hidden)
    np.save('bias_output.npy', nn.bias_output)