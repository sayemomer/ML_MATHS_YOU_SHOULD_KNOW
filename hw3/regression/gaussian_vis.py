import numpy as np
import matplotlib.pyplot as plt
from utils import loadData, normalizeData

[t,X]=loadData()
X_n = normalizeData(X)
t_n = normalizeData(t)

# Extract the 3rd feature
X_n = X_n[:, 2]

# Split the data into training and test sets
X_train = X_n[:100]
t_train = t_n[:100]
X_test = X_n[100:]
t_test = t_n[100:]

# Gaussian Kernel Function
# x1 and x2 are 2D arrays with shape (n, d) and (m, d) respectively
# Returns a 1D array with shape (n,) containing the kernel values
def gaussian_kernel(x1, x2, bandwidth):
    return np.exp(-np.sum((x1 - x2) ** 2, axis=1) / (2 * bandwidth ** 2))

# Kernel Regression Function
# X_train and X_test are 2D arrays with shape (n, d)
# t_train is a 1D array with shape (n,)
# bandwidth is a scalar
# Returns a 1D array with shape (m,) containing the predicted target values
def kernel_regression(X_train, t_train, X_test, bandwidth):
    predictions = np.zeros(X_test.shape[0])

    # Iterate over all test points
    for i, x in enumerate(X_test):
        # Calculate the kernel weights for each training point
        weights = gaussian_kernel(X_train, np.full(X_train.shape, x), bandwidth)
        
        # Weighted average of target values
        predictions[i] = np.average(t_train, weights=weights)
    
    return predictions

# Test the kernel regression function with different bandwidth values

bandwidth_values = [0.01, 0.1, 1, 2, 3, 4]

# Iterate over all bandwidth values
# and visualize the predictions
# for each bandwidth
# The red points represent the predictions
# The blue points represent the training data
# The green points represent the test data
# The x-axis represents the feature values
for h in bandwidth_values:
    t_pred = kernel_regression(X_train.reshape(-1, 1), t_train, X_test.reshape(-1, 1), h)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, t_train, color='blue', label='Training Data')
    plt.scatter(X_test, t_test, color='green', label='Test Data')
    plt.scatter(X_test, t_pred, color='red', label='Predictions')
    plt.title(f'Kernel Regression with bandwidth = {h}')
    plt.legend()
    plt.show()



