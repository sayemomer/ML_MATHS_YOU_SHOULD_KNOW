import numpy as np
import matplotlib.pyplot as plt
from utils import loadData, normalizeData


[t, X] = loadData()
X_n = normalizeData(X)
t_n = normalizeData(t)


X_n = X_n[:, 2] 

X_train = X_n[:100]
t_train = t_n[:100]
X_test = X_n[100:]
t_test = t_n[100:]

# Epanechnikov Kernel Function
# The Epanechnikov kernel is defined as K(x1, x2) = 0.75 * (1 - u^2) if |u| <= 1, and 0 otherwise
# where u = (x1 - x2) / bandwidth
def epanechnikov_kernel(x1, x2, bandwidth):
    u = (x1 - x2) / bandwidth
    return 0.75 * (1 - u**2) * (abs(u) <= 1)

# Kernel Regression Function with Epanechnikov kernel
# where the bandwidth is a hyperparameter that controls the width of the kernel
def kernel_regression(X_train, t_train, x, bandwidth, epsilon=1e-8):
    weights = epanechnikov_kernel(X_train, x, bandwidth)
    weight_sum = np.sum(weights)
    if weight_sum < epsilon:
        return np.nan  # or handle this case in a way that fits your problem
    return np.sum(weights * t_train) / (weight_sum + epsilon)

# Define the bandwidth values
bandwidth_values = [0.01, 0.1, 1, 2, 3, 4]

# Perform kernel regression with the Epanechnikov kernel and visualize the result
# for each bandwidth value
# The red points represent the predictions
# The blue points represent the training data
# The green points represent the test data
for h in bandwidth_values:
    predictions = np.array([kernel_regression(X_train, t_train, x, h) for x in X_test])
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, t_train, color='blue', label='Training Data')
    plt.scatter(X_test, t_test, color='green', label='Test Data')
    plt.scatter(X_test, predictions, color='red', label='Predictions with h=' + str(h))
    plt.xlabel('Feature Value')
    plt.ylabel('Target')
    plt.title(f'Kernel Regression with Epanechnikov Kernel (h={h})')
    plt.legend()
    plt.show()
