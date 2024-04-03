import numpy as np
import matplotlib.pyplot as plt

# Generating a sample dataset
np.random.seed(42)
X = np.random.rand(10,2) * 10  # 10 points in 2D space

# Initial centroids based on points A and B
C1 = np.array([3.75, 9.51])  # Centroid 1
C2 = np.array([7.32, 5.99])  # Centroid 2

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def update_medoids(X, assignments):
    C1_index, C2_index = 0, 0
    min_distance_C1, min_distance_C2 = np.inf, np.inf
    
    for i in range(X.shape[0]):
   
        if assignments[i] == 0:
            # Calculate total distance of all points assigned to C1
            total_distance = np.sum([euclidean_distance(X[i], point) for point in X[assignments == 0]])
            if total_distance < min_distance_C1:
                min_distance_C1 = total_distance
                C1_index = i
        else:
            total_distance = np.sum([euclidean_distance(X[i], point) for point in X[assignments == 1]])
            if total_distance < min_distance_C2:
                min_distance_C2 = total_distance
                C2_index = i
                
    return X[C1_index], X[C2_index]

# Initialize plot
plt.ion()  # Turn on interactive mode
figure, ax = plt.subplots(figsize=(8, 6))

for iteration in range(10):  # Limit the iterations to prevent infinite loop
    assignments = np.zeros(X.shape[0])
    for i, point in enumerate(X):
        distance_to_C1 = euclidean_distance(point, C1)
        distance_to_C2 = euclidean_distance(point, C2)
        assignments[i] = 0 if distance_to_C1 < distance_to_C2 else 1
    
    # Clear the previous plot
    ax.clear()
    
    # Plot data points and centroids with current assignments
    ax.scatter(X[:, 0], X[:, 1], c=assignments, cmap='viridis', marker='o', label='Data Points')
    ax.scatter([C1[0], C2[0]], [C1[1], C2[1]], c='red', s=200, alpha=0.5, marker='X', label='Centroids')
    plt.title(f'K-means Clustering Iteration {iteration+1}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)  # Pause to visualize update
    
    # Update centroids
    C1_new, C2_new = update_medoids(X, assignments)
    
    # Check for convergence
    if np.allclose(C1, C1_new) and np.allclose(C2, C2_new):
        print(f"Converged after {iteration+1} iterations")
        break
    
    C1, C2 = C1_new, C2_new

plt.ioff()  # Turn off interactive mode
plt.show()
