import numpy as np

# Logistic function (also known as the sigmoid function)
def logistic_function(x):
    return 1 / (1 + np.exp(-x))

# Example: Find the value of the logistic function at x = 0.5
x_value = 100
logistic_value = logistic_function(x_value)
print(f"The value of the logistic function at x = {x_value} is {logistic_value}")