import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
digits_train = pd.read_csv('data/train_data.csv')
digits_test  = pd.read_csv('data/emnist-letters-test.csv')

# Set the training data
data_train = np.array(digits_train.T)
Y_train    = data_train[0]   # These are the labels
X_train    = data_train[1:]  # These are the actual observations
X_train    = X_train / 255   # Normalize so each entry is between 0 and 1
_, m_train  = X_train.shape  # Get the number of observations

# Set the test data
data_test = np.array(digits_test.T)
Y_dev     = data_test[0]  # These are the labels
X_dev     = data_test[1:] # These are the actual observations
X_dev     = X_dev / 255   # Normalize so each entry is between 0 and 1

# Initialize parameters
def init_params():
    W1 = np.random.randn(64, 784) * 0.01
    b1 = np.zeros((64, 1))
    W2 = np.random.randn(32, 64) * 0.01
    b2 = np.zeros((32, 1))
    W3 = np.random.randn(26, 32) * 0.01
    b3 = np.zeros((26, 1)) 
    return W1, b1, W2, b2, W3, b3

# Forward propagation
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_propagation(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Backward propagation
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 26))      # Create a matrix of size m x 26
    one_hot_Y[np.arange(Y.size), Y - 1] = 1 # Adjust labels from 1-26 to 0-25 for 0-based indexing
    one_hot_Y = one_hot_Y.T                 # Make each column an observation, instead of the rows
    return one_hot_Y

def sigmoid_deriv(A):
    return A * (1 - A)

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / X.shape[1] * dZ3.dot(A2.T)
    db3 = 1 / X.shape[1] * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = W3.T.dot(dZ3)
    dZ2 = dA2 * sigmoid_deriv(A2)
    dW2 = 1 / X.shape[1] * dZ2.dot(A1.T)
    db2 = 1 / X.shape[1] * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * sigmoid_deriv(A1)
    dW1 = 1 / X.shape[1] * dZ1.dot(X.T)
    db1 = 1 / X.shape[1] * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1    
    W2 -= alpha * dW2  
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3

# Loss function
def compute_loss(A3, Y):
    one_hot_Y = one_hot(Y)
    loss = -np.mean(one_hot_Y * np.log(A3))
    return loss

# Gradient descent with mini-batch
def gradient_descent(X, Y, alpha, iterations, batch_size):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        permutation = np.random.permutation(X.shape[1])
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[permutation]

        for j in range(0, X.shape[1], batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]

            Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        
        if i % 10 == 0:
            Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
            loss = compute_loss(A3, Y)
            print(f"Iteration: {i}, Loss: {loss}")
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y)
            print(f"Accuracy: {accuracy}")
            
        # Learning rate schedule
        if i % 50 == 0 and i != 0:
            alpha *= 0.9  # Reduce learning rate by 10% every 30 iterations
                          # to prevent overshooting at latter stages of training
            
    return W1, b1, W2, b2, W3, b3

# Prediction functions
def get_predictions(A):
    return np.argmax(A, 0) + 1  # Adjust the predictions to be in the range 1-26 to match the labels

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Train the model with mini-batch gradient descent
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.3, 200, batch_size=64)

# Test the model
test_prediction(0, W1, b1, W2, b2, W3, b3)
test_prediction(1, W1, b1, W2, b2, W3, b3)
test_prediction(2, W1, b1, W2, b2, W3, b3)
test_prediction(3, W1, b1, W2, b2, W3, b3)

# Evaluate accuracy on dev set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3)
print("Dev set accuracy:", get_accuracy(dev_predictions, Y_dev))