import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt
from utils import *
import time

# Maximum number of iterations. Continue until this limit, or when erro change is below tol.
max_iter = 1000
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

print('Initial w:',w)

# Error values over all iterations
e_all = np.array([])

# Set up the slope-intercept figure
plt.figure(2)
plt.rcParams['font.size']=20
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])

λ = [0.1, 1, 10, 100]

# negative log-likelihood

# step size 0.003
#for λ, 0.1 = 58.9790
#         1 = 150.0921
#       10 = 1047.3273
#       100 = 10172.5345

# step size 0.001
#for λ, 0.1 = 62.8079
#       1 = 81.7519
#       10 = 108.8628
#       100 = 2438.6068

# for iterations 600
#for step size 0.001
#for λ, 0.1 = 60.9000

# for iterations 1000
#it stops at 741 iterations
#for step size 0.001
#for λ , 0.1 = 59.0988
#norm of w: 5.95862094204972




for iter in range(max_iter):

    # Shuffle the data at the beginning of each epoch
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    t = t[indices]
    # Compute output using current w on all data X.
    y = sigmoid(w.T @ X.T).T

    # e is the rror, negative log-likelihood
    e = -np.sum(t * np.log(y) + (1-t) * np.log(1-y)) + λ[0] * np.sum(w**2)

    # Add this error to the end of error vector
    e_all = np.append(e_all, e)

    # Gradient of the error, using Eqn 4.91
    grad_e = np.sum((y-t)*X, 0, keepdims=True) # 1-by-3
          
    # Update w, *subtracking* a step in the error derivative since we are minimizing
    w_old = w
    w = w - eta*grad_e.T

    if 1:
        # Plot current separator and data
        plt.figure(1)
        plt.clf()
        plt.rcParams['font.size']=20
        plt.plot(X1[:,0],X1[:,1],'g.')
        plt.plot(X2[:,0],X2[:,1],'b.')
        drawSep(plt,w)
        plt.title('Separator in data space')
        plt.axis([-5,15,-10,10])
        plt.draw()
        plt.pause(1e-17)

    # Add next step of separator in m-b space
    plt.figure(2)
    plotMB(plt,w,w_old)
    plt.draw()
    plt.pause(1e-17)

    # Print some information
    print('iter %d, negative log-likelihood %.4f, w=%s' % (iter,e,np.array2string(w.T)))
    #print norm of w
    print('norm of w:',np.linalg.norm(w))

    # Stop iterating if error does not change more than tol
    if iter > 0:
        if abs(e-e_all[iter-1]) < tol:
            break


    
# Plot error over iterations
plt.figure(3)
plt.rcParams['font.size']=20
plt.plot(e_all,'b-')
plt.xlabel('Iteration')
plt.ylabel('neg. log likelihood')
plt.title('Minimization using gradient descent')

plt.show()
