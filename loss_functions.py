import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute the cross-entropy loss.
    
    Parameters:
    y_true (numpy.ndarray): True labels (one-hot encoded).
    y_pred (numpy.ndarray): Predicted probabilities.
    
    Returns:
    float: Cross-entropy loss.
    """
    n_samples = y_true.shape[0]
    y_true = np.argmax(y_true, axis=1)  # Convert one-hot encoded labels to class labels
    res = y_pred[range(n_samples), y_true]
    log_likelihood = -np.log(res + 1e-9)  # Add a small value to avoid log(0)
    loss = np.sum(log_likelihood) / n_samples
    return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    """
    Compute the derivative of the cross-entropy loss.
    
    Parameters:
    y_true (numpy.ndarray): True labels (one-hot encoded).
    y_pred (numpy.ndarray): Predicted probabilities.
    
    Returns:
    numpy.ndarray: Derivative of the loss with respect to y_pred.
    """
    n_samples = y_true.shape[0]
    y_true = np.argmax(y_true, axis=1)  # Convert one-hot encoded labels to class labels
    grad = y_pred.copy()
    grad[range(n_samples), y_true] -= 1
    grad = grad / n_samples
    return grad
