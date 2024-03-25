import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import sigmoid, drawSep, plotMB  # Make sure these are implemented correctly

# Parameters
max_iter = 500
tol = 0.01
eta = 0.003

# Load and prepare data
data = loadmat('/Users/sayems_mac/ml6321/logistic/data.mat')  # Update this path as needed
X1, X2 = data['X1'], data['X2']
X = np.vstack((X1, X2))
X = np.hstack((X, np.ones((X.shape[0], 1))))  # Add a column of ones for the bias term
t = np.vstack((np.zeros((X1.shape[0], 1)), np.ones((X2.shape[0], 1))))

# Initialize weights
w = np.array([1., 0., 0.]).reshape(3, 1)

# Initialize error tracking
e_all = []

# Set up the slope-intercept figure

plt.figure(2)
plt.rcParams['font.size'] = 20
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])

# Perform SGD
for iter in range(max_iter):
    for i in range(X.shape[0]):
        # Compute output using current w for one data point X[i]
        y = sigmoid(np.dot(w.T, X[i]))

        # Compute error (negative log-likelihood) for one data point
        e = -(t[i] * np.log(y) + (1 - t[i]) * np.log(1 - y))

        # Compute gradient for one data point
        grad_e = (y - t[i]) * X[i]

        w_old = w

        # Update weights
        w -= eta * grad_e.reshape(w.shape)

    # Compute error over all data points at the end of epoch
    y_all = sigmoid(X @ w)
    e_epoch = -np.sum(t * np.log(y_all) + (1 - t) * np.log(1 - y_all))
    e_all.append(e_epoch)

    if 1:
        # Plot separator and data
        plt.figure(1)
        plt.clf()
        plt.rcParams['font.size'] = 20
        plt.plot(X1[:, 0], X1[:, 1], 'g.')
        plt.plot(X2[:, 0], X2[:, 1], 'b.')
        drawSep(plt, w)
        plt.title('Separator in data space')
        plt.axis([-5, 15, -10, 10])
        plt.draw()
        plt.pause(1e-17)
    
    plt.figure(2)
    plotMB(plt, w,w_old)
    plt.draw()
    plt.pause(1e-17)


    # Convergence check at the end of each epoch
    if iter > 0 :
        if np.abs(e-e_all[iter-1]) < tol:
            print(f'Converged at iter: {iter+1}, Negative Log Likelihood: {e_all[-1]}')
            break


plt.figure(3)
plt.rcParams['font.size'] = 20
plt.plot(e_all,'b-')
plt.xlabel('Iteration')
plt.ylabel('Negative log likelihood')
plt.title('Minimization using Gradient Descent')

plt.show()
