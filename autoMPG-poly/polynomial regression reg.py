import numpy as np
import matplotlib.pyplot as plt

import utils as u

# Load the data
target, X = u.loadData()

# Normalize the data
normalized_X = u.normalizeData(X)
normalized_t = u.normalizeData(target)


# regularized linear regression function
def reglinearRegression(X, y, lambda_val):
    I = np.eye(X.shape[1])  # Identity matrix for regularization
    w = np.linalg.inv(X.T @ X + lambda_val * I) @ X.T @ y
    return w

def predict(X, w):
    return X @ w

def mse(t, t_pred):
    return np.mean((t - t_pred) ** 2)

# Perform 10-fold cross-validation
def kFoldCrossValidation(k, X, y, lambda_values):
    fold_size = len(X) // k
    avg_validation_errors = []
    
    for lambda_val in lambda_values:
        validation_errors = []
        for fold in range(k):
            # Create validation set and training set
            start_val = fold * fold_size
            end_val = start_val + fold_size
            if fold == k - 1:
                end_val = len(X)  # Ensure the last fold includes the remainder
            
            X_val = X[start_val:end_val]
            y_val = y[start_val:end_val]
            X_train = np.concatenate((X[:start_val], X[end_val:]), axis=0)
            y_train = np.concatenate((y[:start_val], y[end_val:]), axis=0)
            
            # Train the model using the training set and the current lambda
            w = reglinearRegression(X_train, y_train, lambda_val)
            
            # Predict on the validation set
            t_pred_val = predict(X_val, w)
            
            # Calculate the validation error for the current fold
            validation_errors.append(mse(y_val, t_pred_val))
        
        # Calculate the average validation error for the current lambda
        avg_validation_errors.append(np.mean(validation_errors))
    
    return avg_validation_errors

# Define the degree and lambda values
degree = 8
lambda_values = [0, 0.01, 0.1, 1, 10, 100, 1000]

# Prepare the data
X_train = normalized_X[:100, 2].reshape(-1, 1)
t_train = normalized_t[:100]

# Expand to degree 8 polynomial
X_poly = u.degexpand(X_train, degree)

# Perform 10-fold cross-validation
avg_validation_errors = kFoldCrossValidation(10, X_poly, t_train, lambda_values)

# Plot the results
plt.semilogx(lambda_values, avg_validation_errors, base=10)
plt.xlabel('Lambda')
plt.ylabel('Average Validation Error')
plt.title('Validation Error vs Regularizer Value')
plt.show()


