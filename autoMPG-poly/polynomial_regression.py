import numpy as np
import utils as u
import matplotlib.pyplot as plt

target, X = u.loadData()

normalized_X = u.normalizeData(X)
normalized_t= u.normalizeData(target)

# Using the first 100 points as training data, and the remainder as testing data
X_train = normalized_X[:100]
t_train = normalized_t[:100]
X_test = normalized_X[100:]
t_test = normalized_t[100:]

# Linear regression function
def linearRegression(X, y):
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w


#regression for degree 1 to degree 10 polynomials
degree = 11

# expand the training data to include the polynomial terms
X_poly = u.degexpand(X_train, degree)

# find the weights for the polynomial

w = linearRegression(X_poly, t_train)

# Predict function
def predict(X, w):
    return X @ w

# Predict the test data
X_test_poly = u.degexpand(X_test, degree)
t_pred = predict(X_test_poly, w)

# find the mean squared error
def mse(t, t_pred):
    return np.mean((t - t_pred) ** 2)


# main mathod

if __name__ == "__main__":
#Plot training error and test error (i.e. mean squared error) versus polynomial degree
    train_error = []
    test_error = []

    for i in range(1, degree):
        X_poly = u.degexpand(X_train, i)
        w = linearRegression(X_poly, t_train)
        t_pred = predict(X_poly, w)
        train_error.append(mse(t_train, t_pred))

        X_poly = u.degexpand(X_test, i)
        t_pred = predict(X_poly, w)
        test_error.append(mse(t_test, t_pred))

    plt.plot(range(1, degree), train_error, label='Training Error')
    plt.plot(range(1, degree), test_error, label='Test Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()