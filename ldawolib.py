import numpy as np

# Define dataset (3 samples, 2 features)
X = np.array([[2, 3], [5, 8], [7, 10]])  # Features
y = np.array([0, 1, 1])  # Class labels

# Step 1: Compute mean vectors for each class
class_labels = np.unique(y)
mean_vectors = {label: np.mean(X[y == label], axis=0) for label in class_labels}

# Step 2: Compute within-class scatter matrix Sw
S_W = np.zeros((2, 2))
for label in class_labels:
    class_scatter = np.zeros((2, 2))
    for row in X[y == label]:  # Select rows belonging to the class
        row, mean_vec = row.reshape(2, 1), mean_vectors[label].reshape(2, 1)
        class_scatter += (row - mean_vec) @ (row - mean_vec).T
    S_W += class_scatter

# Step 3: Compute between-class scatter matrix Sb
overall_mean = np.mean(X, axis=0).reshape(2, 1)
S_B = np.zeros((2, 2))
for label in class_labels:
    n = X[y == label].shape[0]  # Number of samples in class
    mean_vec = mean_vectors[label].reshape(2, 1)
    S_B += n * (mean_vec - overall_mean) @ (mean_vec - overall_mean).T

# Step 4: Solve for eigenvalues and eigenvectors of Sw^-1 * Sb using pseudo-inverse
eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))  # FIXED: Use pinv()

# Step 5: Select the eigenvector with the highest eigenvalue
top_eigvec = eig_vecs[:, np.argmax(eig_vals)].reshape(2, 1)

# Step 6: Project the data onto the new LDA axis
X_lda = X.dot(top_eigvec)

# Output the transformed data
print("Transformed Data (LDA):\n", X_lda.real)
