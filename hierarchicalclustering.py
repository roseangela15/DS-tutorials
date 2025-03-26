import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs

# Step 1: Generate Sample Data
X, _ = make_blobs(n_samples=10, centers=3, random_state=42)

# Step 2: Compute the linkage matrix using complete linkage
linkage_matrix = sch.linkage(X, method='complete')

# Step 3: Plot the dendrogram
plt.figure(figsize=(8, 5))
sch.dendrogram(linkage_matrix, labels=np.arange(len(X)))
plt.title("Hierarchical Clustering (Complete Linkage)")
plt.xlabel("Data Point Index")
plt.ylabel("Distance")
plt.show()
