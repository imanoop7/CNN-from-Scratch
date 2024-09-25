import numpy as np
from neural_network import NeuralNetwork
from data_loader import load_data

if __name__ == "__main__":
    print("Loading data...")
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    print(f"Data loaded successfully.")
    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # Convert labels to one-hot encoding
    print("Converting labels to one-hot encoding.")
    y_train_one_hot = np.zeros((y_train.size, 10))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1
    print(f"One-hot encoded labels shape: {y_train_one_hot.shape}")
    
    # Network parameters
    epochs = 10  # Reduced for faster initial testing
    learning_rate = 0.01
    batch_size = 128  # Added batch size for mini-batch training
    print(f"Training parameters - Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}")
    
    # Initialize and train the network
    print("Initializing Neural Network.")
    nn = NeuralNetwork()
    print("Starting training.")
    nn.train(X_train, y_train_one_hot, epochs, learning_rate, batch_size)
    print("Training process finished.")
    
    # Save the trained weights and biases
    print("Saving trained weights and biases.")
    np.save('conv_filters.npy', nn.conv.filters)
    print("Saved conv_filters.npy")
    np.save('conv_biases.npy', nn.conv.biases)
    print("Saved conv_biases.npy")
    np.save('fc_weights.npy', nn.fc.weights)
    print("Saved fc_weights.npy")
    np.save('fc_bias.npy', nn.fc.bias)
    print("Saved fc_bias.npy")
    
    print("Training completed and weights saved.")