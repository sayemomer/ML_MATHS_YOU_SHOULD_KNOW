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

# Gaussian Kernel Function
# The Gaussian kernel is defined as K(x1, x2) = exp(-(x1 - x2)^2 / (2 * bandwidth^2))
def gaussian_kernel(x1, x2, bandwidth):
    return np.exp(-(x1 - x2) ** 2 / (2 * bandwidth ** 2))

# Kernel Regression Function with epsilon correction
# where the bandwidth is a hyperparameter that controls the width of the kernel
# and epsilon is a small value to avoid division by zero
def kernel_regression(X_train, t_train, x, bandwidth, epsilon=1e-8):
    weights = gaussian_kernel(X_train, x, bandwidth)
    weight_sum = np.sum(weights)
    # If the sum of weights is zero or extremely small, add epsilon to avoid division by zero
    if weight_sum < epsilon:
        return np.nan  # or handle this case in a way that fits your problem
    return np.sum(weights * t_train) / (weight_sum + epsilon)

# 10-fold cross-validation
def cross_validate(X, t, bandwidth_values):
    # Split the data into 10 folds
    fold_size = len(X) // 10
    errors = {h: [] for h in bandwidth_values}
    

    # Perform 10-fold cross-validation
    # For each fold, calculate the error for each bandwidth value
    for fold in range(10):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_valid, t_valid = X[start:end], t[start:end]
        X_train = np.concatenate((X[:start], X[end:]))
        t_train = np.concatenate((t[:start], t[end:]))

        # Calculate the error for each bandwidth value
        # If all predictions are NaN, assign a high error to the bandwidth
        # This can happen if the sum of weights is zero or extremely small
        for h in bandwidth_values:
            fold_errors = []
            predictions = np.array([kernel_regression(X_train, t_train, x, h) for x in X_valid])
            # Filter out NaN values from predictions before calculating the error
            valid_predictions = predictions[~np.isnan(predictions)]
            valid_t_valid = t_valid[~np.isnan(predictions)]
            if len(valid_predictions) > 0:
                fold_errors.append(np.mean((valid_predictions - valid_t_valid) ** 2))
            else:
                fold_errors.append(np.inf) 
            errors[h].append(fold_errors)

    # Calculate the average of the errors for each bandwidth
    average_errors = {h: np.mean(errors[h]) for h in bandwidth_values}
    return average_errors

# Define the bandwidth values
bandwidth_values = [0.01, 0.1, 0.25, 1, 2, 3, 4]

# Perform 10-fold cross-validation for each bandwidth
average_errors = cross_validate(X_train, t_train, bandwidth_values)

# Plot the validation error against bandwidth values on a semilogx scale
plt.figure(figsize=(10, 6))
plt.semilogx(bandwidth_values, list(average_errors.values()), marker='o', linestyle='-')
plt.xlabel('Bandwidth h')
plt.ylabel('Average Validation Error')
plt.title('10-fold Cross-Validation: Error vs. Bandwidth h')
plt.grid(True)
plt.show()

# Select the best bandwidth based on the lowest average validation error
best_bandwidth = min(average_errors, key=average_errors.get)
print(f"Average validation errors for each bandwidth: {average_errors}")
print(f"The best bandwidth h value from cross-validation is: {best_bandwidth}")
