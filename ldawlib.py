import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Adding an extra sample (at least 3 samples needed)
X = np.array([[2, 3], [5, 8], [7, 10]])  # Now 3 samples, 2 features
y = np.array([0, 1, 1])  # 2 classes (0 and 1), but 3 samples

# Initialize LDA model
lda = LinearDiscriminantAnalysis(n_components=1)  # Reduce to 1 component

# Fit LDA model
lda.fit(X, y)

# Transform the data
X_lda = lda.transform(X)

print("Transformed Data (LDA):\n", X_lda)
