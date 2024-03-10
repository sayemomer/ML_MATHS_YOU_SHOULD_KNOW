import numpy as np;


data_points = [1, 2, 3.5, 4, 5]

class_labels = [-1, -1, -1, 1, 1]

w_val = 14



# Function to calculate the decision boundary based on remaining points
def calculate_decision_boundary(left_out_point, data_points, class_labels):
    # Exclude the left out point from the dataset
    remaining_points = [x for i, x in enumerate(data_points) if x != left_out_point]
    remaining_labels = [y for i, y in enumerate(class_labels) if data_points[i] != left_out_point]
    
    # Find the support vectors among the remaining points
    support_vectors = [remaining_points[i] for i, y in enumerate(remaining_labels) if y == -1][-1], \
                      [remaining_points[i] for i, y in enumerate(remaining_labels) if y == 1][0]
    
    # Calculate the mid-point between support vectors
    decision_boundary = (support_vectors[0] + support_vectors[1]) / 2
    return decision_boundary

# Check if the left-out point would be misclassified
def is_misclassified(left_out_point, decision_boundary, true_label, w):
    predicted_label = np.sign(w * left_out_point - w * decision_boundary)
    return predicted_label != true_label

# Leave-one-out cross-validation process
misclassifications = 0

for point, label in zip(data_points, class_labels):
    decision_boundary = calculate_decision_boundary(point, data_points, class_labels)
    if is_misclassified(point, decision_boundary, label, w_val):
        misclassifications += 1

# LOOCV error as a percentage
loocv_error = (misclassifications / len(data_points)) * 100
print(loocv_error)
