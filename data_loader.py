import numpy as np
import tensorflow as tf

def load_data():
    print("Loading MNIST dataset from TensorFlow.")
    # Load MNIST dataset from TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("MNIST dataset loaded successfully.")
    
    # Normalize the images to the range [0, 1]
    print("Normalizing images to [0, 1].")
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Reshape the images for CNN input (add channel dimension)
    print("Reshaping images for CNN input.")
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    print(f"Reshaped X_train shape: {X_train.shape}")
    print(f"Reshaped X_test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test