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
    epochs = 1000
    learning_rate = 0.01
    
    # Initialize and train the network
    nn = NeuralNetwork()
    nn.train(X_train, y_train_one_hot, epochs, learning_rate)
    
    # Save the trained weights and biases
    np.save('conv_filters.npy', nn.conv.filters)
    np.save('conv_biases.npy', nn.conv.biases)
    np.save('fc_weights.npy', nn.fc.weights)
    np.save('fc_bias.npy', nn.fc.bias)