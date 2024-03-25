import numpy as np

# Define a simple dataset
X = np.array([-2, -1, 1, 2])  # Features
y = np.array([-1, -1,1, 1])  # Labels

# Initialize weights
weights = np.ones(len(X)) / len(X)

# Number of iterations
T = 2

def train_stump(X, y, weights):
    """Trains a decision stump by finding the best threshold"""
    best_error = float('inf')
    best_stump = {'threshold': 0, 'direction': 1}
    
    for threshold in X:
        for direction in [-1, 1]:
            print(f"Threshold: {threshold}, Direction: {direction}")
            predictions = np.ones(len(X))  # Initialize predictions to 1
            # Flip predictions based on threshold and direction
            # If direction is -1, flip predictions where X is less than or equal to threshold
            # If direction is 1, flip predictions where X is greater than or equal to threshold
            # This is equivalent to the following line:
            # predictions = np.sign(direction * (X - threshold))
            predictions[direction * X <= direction * threshold] = -1  # Flip predictions based on direction and threshold
            print(predictions)
            error = sum(weights[predictions != y])
            print(error)
            
            # Ensure error is within bounds to avoid division by zero or log of zero
            error = max(min(error, 1 - 1e-10), 1e-10)
            
            if error < best_error:
                best_error = error
                best_stump['threshold'] = threshold
                best_stump['direction'] = direction
    
    return best_stump, best_error

alphas = []
stumps = []

for t in range(T):
    # Train a decision stump using weighted samples
    stump, error = train_stump(X, y, weights)
    print(f"Iteration {t + 1}: Stump: {stump}, Error: {error}")

    #Step 1: Calculate alpha
    # Calculate alpha, using a small offset to prevent division by zero
    alpha = 0.5 * np.log((1 - error) / error)
    
    # Update sample weights
    predictions = np.ones(len(X))
    predictions[stump['direction'] * X <= stump['direction'] * stump['threshold']] = -1
    print(f"pred: {predictions}")

    # Step 2: Update weights
    weights *= np.exp(-alpha * y * predictions)
    weights /= np.sum(weights)  # Normalize weights
    print(f"Weights: {weights}")
    
    # Save the stump and its alpha
    alphas.append(alpha)
    stumps.append(stump)

# Make final predictions
final_scores = np.zeros(len(X))
for alpha, stump in zip(alphas, stumps):
    predictions = np.ones(len(X))
    predictions[stump['direction'] * X <= stump['direction'] * stump['threshold']] = -1
    # step 3: Combine predictions
    final_scores += alpha * predictions

final_predictions = np.sign(final_scores)

# Output the final predictions
print("Final Predictions:", final_predictions)
