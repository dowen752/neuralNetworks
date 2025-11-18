import numpy as np
import neuralMethods as nm

# Define network architecture
input_size = 2 # XOR takes 2 inputs
hidden_size = 4 # Number of neurons in hidden layer
output_size = 2 # Either 1 or 0 for XOR

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]]) # X is our inputs for each XOR
y = np.array([[1,0],[0,1],[0,1],[1,0]]) # y is our expected outputs. 
                                        # First value is for class 0 (false), 
                                        # second for class 1 (true)

lr = 0.1
epochs = 10000 # Really overkill for XOR, reaches desired prediction by like 500 epochs.
               # Past this point, it just loops towards a predicatable state.

for epoch in range(epochs):
    # ---- Forward pass ----
    z1 = X.dot(W1) + b1 # weighted sum for hidden layer (dot product of X rows and W1 columns)
    a1 = nm.sigmoid(z1) # activation for hidden layer by sigmoid
    z2 = a1.dot(W2) + b2 # weighted sum for output layer (dot produt of a1 rows and W2 columns)
    y_pred = nm.softmax(z2) # activation for output layer by softmax

    # ---- Loss ----
    cost = nm.cross_entropy_cost(y, y_pred) # comparing predicted output (y_pred) with known output (y)

    # ---- Backward pass ----
    dC_dz2 = y_pred - y # Gradient of cost w.r.t. z2; although this is a derivative, it simplifies to just (y_pred - y) for softmax + cross-entropy
    dC_dW2 = a1.T.dot(dC_dz2) # Gradient w.r.t. W2
    dC_db2 = np.sum(dC_dz2, axis=0, keepdims=True) # Gradient w.r.t. b2

    dC_da1 = dC_dz2.dot(W2.T) # derivative of loss w.r.t. a1
    dC_dz1 = dC_da1 * nm.sigmoid_prime(a1) # derivative of loss w.r.t. z1
    dC_dW1 = X.T.dot(dC_dz1) # Gradient w.r.t. W1
    dC_db1 = np.sum(dC_dz1, axis=0, keepdims=True) # Gradient w.r.t. b1

    # The gradients dC_dW1, dC_db1, dC_dW2, dC_db2 show us how sensitive the cost is to changes in each weight and bias.
    
    # ---- Parameter update ----
    # Update weights and biases using gradient descent
    W1 -= lr * dC_dW1
    b1 -= lr * dC_db1
    W2 -= lr * dC_dW2
    b2 -= lr * dC_db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Cost: {cost:.4f}")

def main():
    
    predictions = np.argmax(y_pred, axis=1)
    print("Predictions:", predictions)
    
if __name__ == "__main__":
    main()
