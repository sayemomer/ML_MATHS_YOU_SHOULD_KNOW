# only use one of the features (use the 3rd feature, i.e. X n[:,2] since Python index starts with 0)
from visualize_1d import visualize_1d
import numpy as np
import utils as u
from polynomial_regression import linearRegression, predict

target, X = u.loadData()

normalized_X = u.normalizeData(X)
normalized_t= u.normalizeData(target)

X_train = normalized_X[:100, 2].reshape(-1, 1)
t_train = normalized_t[:100]
X_test = normalized_X[100:, 2].reshape(-1, 1)
t_test = normalized_t[100:]

# again perform polynomial regression for degree 1 to degree 10 polynomials
degree = 11

X_poly = u.degexpand(X_train, degree)

# find the weights for the polynomial

w = linearRegression(X_poly, t_train)

# Predict the test data
X_test_poly = u.degexpand(X_test, degree)
t_pred = predict(X_test_poly, w)

# Call the visualize_1d function with the appropriate parameters
visualize_1d(X_train.flatten(), X_test.flatten(), t_train, t_test, degree, w)



