import numpy as np

# Generate sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Polynomial degree
degree = 5

# Polynomial feature transformation
def polynomial_features(x, degree):
    # For each element in x, create a feature vector [1, x, x^2, ..., x^degree]
    return np.array([[xi**d for d in range(degree + 1)] for xi in x])

# Ordinary Least Squares Linear Regression
def linear_regression(X, y):
    # Compute the parameters: w = (X^T X)^{-1} X^T y
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

# Transform input data
X_poly = polynomial_features(x, degree)

print("Polynomial features:", X_poly)

# Fit the model
w = linear_regression(X_poly, y)

print("Model coefficients:", w)

# Predict function
def predict(X, w):
    return X @ w

# Test the model
x_test = np.array([6, 7, 8])
X_test_poly = polynomial_features(x_test, degree)
predictions = predict(X_test_poly, w)

print("Predictions:", predictions)
