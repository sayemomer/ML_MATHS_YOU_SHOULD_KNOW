# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate the means of x and y
mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

# Calculate the slope (m)
numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
denominator = sum((xi - mean_x) ** 2 for xi in x)
m = numerator / denominator

# Calculate the intercept (b)
b = mean_y - m * mean_x

print(f"Linear model is: y = {m}x + {b}")

# Predict function
def predict(x):
    return m * x + b

# Make predictions
x_test = [6, 7, 8]
predictions = [predict(xi) for xi in x_test]

print("Predictions:", predictions)
