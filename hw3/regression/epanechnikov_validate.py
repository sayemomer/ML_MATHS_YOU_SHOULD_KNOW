import numpy as np
import matplotlib.pyplot as plt
from utils import loadData, normalizeData


[t, X] = loadData()
X_n = normalizeData(X)
t_n = normalizeData(t)

X_n = X_n[:, 2]


X_train = X_n[:100]
t_train = t_n[:100]

# Epanechnikov Kernel Function
# The Epanechnikov kernel is defined as K(x1, x2) = 0.75 * (1 - u^2) if |u| <= 1, and 0 otherwise
# where u = (x1 - x2) / bandwidth
def epanechnikov_kernel(x1, x2, bandwidth):
    u = (x1 - x2) / bandwidth
    return 0.75 * (1 - u ** 2) * (abs(u) <= 1)

# Kernel Regression Function using Epanechnikov kernel
# where the bandwidth is a hyperparameter that controls the width of the kernel
# and epsilon is a small value to avoid division by zero
# The function returns the predicted target value for a given input x
def kernel_regression(X_train, t_train, x, bandwidth, epsilon=1e-8):
    weights = epanechnikov_kernel(X_train, x, bandwidth)
    weight_sum = np.sum(weights)
    if weight_sum < epsilon:
        return np.nan  # or handle this case in a way that fits your problem
    return np.sum(weights * t_train) / (weight_sum + epsilon)

# 10-fold cross-validation
# This function performs 10-fold cross-validation for each bandwidth value
# and returns the average validation error for each bandwidth
def cross_validate(X, t, bandwidth_values):
    fold_size = len(X) // 10
    errors = {h: [] for h in bandwidth_values}

    for fold in range(10):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_valid, t_valid = X[start:end], t[start:end]
        X_train = np.concatenate((X[:start], X[end:]))
        t_train = np.concatenate((t[:start], t[end:]))

        for h in bandwidth_values:

            predictions = np.array([kernel_regression(X_train, t_train, x, h) for x in X_valid])
            # Filter out NaN values from predictions before calculating the error
            valid_predictions = predictions[~np.isnan(predictions)]
            valid_t_valid = t_valid[~np.isnan(predictions)]
            if len(valid_predictions) > 0:
                errors[h].append(np.mean((valid_predictions - valid_t_valid) ** 2))
            else:
                errors[h].append(np.inf)  # Assign a high error if all predictions are NaN

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
plt.title('10-fold Cross-Validation with Epanechnikov Kernel: Error vs. Bandwidth h')
plt.grid(True)
plt.show()

# Select the best bandwidth based on the lowest average validation error
best_bandwidth = min(average_errors, key=average_errors.get)
print(f"Average validation errors for each bandwidth: {average_errors}")
print(f"The best bandwidth h value from cross-validation with Epanechnikov Kernel is: {best_bandwidth}")
