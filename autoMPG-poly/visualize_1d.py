import matplotlib.pyplot as plt
import numpy as np

from polynomial_regression import predict
import utils as u

# Assuming u.degexpand and predict functions are defined elsewhere
# Assuming w (weights) are computed from your polynomial regression

def visualize_1d(X_train, X_test, t_train, t_test, degree, w):
    # Generate a range of x values for plotting the regression curve
    X_n = np.linspace(min(np.concatenate([X_train, X_test])), max(np.concatenate([X_train, X_test])), 200).reshape(-1, 1)
    
    # Expand X_n to polynomial features
    X_n_poly = u.degexpand(X_n, degree)
    
    # Predict y values for the generated range of x values
    t_n = predict(X_n_poly, w)
    
    # Plot the regression estimate
    plt.plot(X_n, t_n, 'r.-', label='Learned Polynomial Degree %d' % degree)
    # Plot training and test data
    plt.scatter(X_train, t_train, color='green', marker='x', s=100, label='Training Data')
    plt.scatter(X_test, t_test, color='blue', marker='o', facecolors='none', s=100, label='Test Data')
    
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.title('Polynomial Regression Visualization')
    plt.show()



