import numpy as np

# ----------------------
# This will be a simple progrmam to practice implementing a neural network from scratch
# ----------------------

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return x * (1.0 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_cost(y_true, y_pred):
    n = y_true.shape[0]
    return -(np.sum(y_true * np.log(y_pred + 1e-9)) / n)