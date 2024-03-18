import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt
from utils import *
import time

# Maximum number of iterations. Continue until this limit, or when erro change is below tol.
max_iter = 100000
tol = 0.01

# Step size for gradient descent
eta = 0.001

# Get X1,X2
data=loadmat('/Users/sayems_mac/ml6321/logistic/data.mat')
X1,X2=data['X1'],data['X2']

# Data matrix with column of ones at end.
X = np.vstack((X1,X2))
X = np.hstack((X,np.ones((X.shape[0],1))))
# Target values, 0 for class 1 (datapoints X1), 1 for class 2 (datapoints X2)
t = np.vstack((np.zeros((X1.shape[0],1)),np.ones((X2.shape[0],1))))


# Initialize w.
w = np.array([1., 0., 0.]).reshape(3,1)

# print('Initial w:',w)

# Set up the slope-intercept figure
plt.figure(2)
plt.rcParams['font.size']=20
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])

λ = [0.1, 1, 10, 100]


def logistic_regression(X, t, w, max_iter, tol, eta, λ):

    #keep track of all the negative log-likelihood and norm of w

    result = np.array([])

    # Error values over all iterations
    e_all = np.array([])

    for iter in range(max_iter):

            # Compute output using current w on all data X.
            y = sigmoid(w.T @ X.T).T

            # e is the rror, negative log-likelihood
            e = -np.sum(t * np.log(y) + (1-t) * np.log(1-y)) + λ * np.sum(w**2)

            # Add this error to the end of error vector
            e_all = np.append(e_all, e)

            # Gradient of the error, using Eqn 4.91
            grad_e = np.sum((y-t)*X, 0, keepdims=True) # 1-by-3
                
            # Update w, *subtracking* a step in the error derivative since we are minimizing
            w = w - eta*grad_e.T

            # # Print some information
            # print('iter %d, negative log-likelihood %.4f, w=%s' % (iter,e,np.array2string(w.T)))
            # #print norm of w
            # print('norm of w:',np.linalg.norm(w))

            # Stop iterating if error does not change more than tol
            if iter > 0:
                if abs(e-e_all[iter-1]) < tol:
                    print('converged. iterations:',iter)
                    break
    print('negative log-likelihood %.4f, w=%s' % (e,np.linalg.norm(w)))

# Run the logistic regression
 
print(logistic_regression(X, t, w, max_iter, tol, eta, 100))

