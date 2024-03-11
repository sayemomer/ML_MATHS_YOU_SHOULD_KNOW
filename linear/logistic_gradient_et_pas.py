# Import necessary libraries
from cmath import inf
import numpy as np

# Direct solution using closed-form

class LinearRegression_Closed_Form:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.w = None

    def fit(self):
        # Assuming X already includes a bias term (column of ones)
        self.w = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y

    def predict(self, X):
        return X @ self.w
    
# Solution using gradient descent
    
class LinearRegression_Gradient_Descent:

    def __init__(self,w,b,eta,iterations) -> None:
        self.w = w
        self.b = b
        self.eta = eta
        self.iterations = iterations
    
    def gradient(self,X,Y):
        N = len(X)
        for i in range(self.iterations):
            # Predicted values
            Y_pred = self.w * X + self.b
            
            # Gradient of the cost function w.r.t w and b
            dw = (-2/N) * sum(X * (Y - Y_pred))
            db = (-2/N) * sum(Y - Y_pred)
            
            # Update parameters
            self.w -= self.eta * dw
            self.b -= self.eta * db
            
            # Compute cost
            cost = (1/N) * sum((Y - Y_pred)**2)
            
            # print(f"Iteration {i+1}: w = {self.w}, b = {self.b}, Cost = {cost}")
        return self.w
    def predict(self,X):
        return X * self.w + self.b

def mean_square_error(N,Y,Y_pred):
    cost = (1/N) * sum((Y - Y_pred)**2)
    return cost


#main

if __name__ == "__main__":
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([2, 4, 6, 8, 10])
    w = 0
    b = 0
    eta = 0.01
    iterations = 10
    N = len(X)
    # Closed form solution
    X = np.c_[np.ones(len(X)), X]
    model = LinearRegression_Closed_Form(X,Y)
    model.fit()
    print("Model coefficients:", model.w)
    print("Predictions:", model.predict(X))
    print("Mean Square Error:", mean_square_error(N,Y,model.predict(X)))
    # Gradient Descent solution
    Y= Y.reshape(-1,1)
    model = LinearRegression_Gradient_Descent(w,b,eta,iterations)
    
    print("Model coefficients:", model.gradient(X,Y))
    print("Predictions:", model.predict(X))
    print("Mean Square Error:", mean_square_error(N,Y,model.predict(X)))
    # model.gradient(X,Y)
    # print("Predictions:", model.predict(X))
    # print("Mean Square Error:", mean_square_error(N,Y,model.predict(X)))

