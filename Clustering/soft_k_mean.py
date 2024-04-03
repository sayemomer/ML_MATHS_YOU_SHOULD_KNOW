import numpy as np
import matplotlib.pyplot as plt

# Generating a sample dataset
np.random.seed(42)
X = np.random.rand(10, 2) * 10  # 10 points in 2D space

# Initial centroids
C1 = np.array([3.75, 9.51])  # Centroid 1
C2 = np.array([7.32, 5.99])  # Centroid 2

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def softmax(distances, beta=1.0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(-beta * np.array(distances))
    return e_x / e_x.sum(axis=0)  # axis=0 for column-wise normalization

def update_centroids(X, memberships):
    weights = memberships / np.sum(memberships, axis=0, keepdims=True)
    C1_new = np.dot(weights[:, 0], X) / np.sum(weights[:, 0])
    C2_new = np.dot(weights[:, 1], X) / np.sum(weights[:, 1])
    return C1_new, C2_new

# Beta parameter for soft assignments
beta = 5.0  # Adjust this to see different behaviors

plt.ion()  # Turn on interactive mode
figure, ax = plt.subplots(figsize=(8, 6))

for iteration in range(10):
    distances = np.zeros((X.shape[0], 2))
    for i, point in enumerate(X):
        distances[i, 0] = euclidean_distance(point, C1)
        distances[i, 1] = euclidean_distance(point, C2)
    
    memberships = np.array([softmax(dist) for dist in distances])
    
    ax.clear()
    
    # Plot with soft assignments
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], color=plt.cm.viridis(memberships[i, 0]), marker='o')
    ax.scatter([C1[0], C2[0]], [C1[1], C2[1]], c='red', s=200, alpha=0.5, marker='X', label='Centroids')
    
    plt.title(f'Soft K-means Clustering Iteration {iteration + 1}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)
    
    C1_new, C2_new = update_centroids(X, memberships)
    
    if np.allclose(C1, C1_new) and np.allclose(C2, C2_new):
        print(f"Converged after {iteration + 1} iterations")
        break
    
    C1, C2 = C1_new, C2_new

plt.ioff()
plt.show()
