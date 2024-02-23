import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# For linear regression, we need to add a column of ones to x to account for the intercept
X = np.vstack([np.ones(x.shape[0]), x]).T
print("Input data:", X)

# Ordinary Least Squares Linear Regression function
def linear_regression(X, y):
    # Compute the parameters: w = (X^T X)^{-1} X^T y
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

# Fit the model
w = linear_regression(X, y)

print("Model coefficients:", w)

# Predict function
def predict(X, w):
    return X @ w

# To test the model, prepare the test data in the same way as the training data
x_test = np.array([6, 7, 8])
X_test = np.vstack([np.ones(x_test.shape[0]), x_test]).T

# Make predictions
predictions = predict(X_test, w)

print("Predictions:", predictions)
