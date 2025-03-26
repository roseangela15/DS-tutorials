import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class RandomForestRegressorCustom:
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[sample_indices], y[sample_indices]
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)  # Averaging ensemble

# Load the Advertising dataset
data = pd.read_csv("C:/Users/Rose Angela Vinod/Desktop/DS TUTORIALS/Tutorial4/advertising.csv")
X = data[["TV", "Radio", "Newspaper"]].values
y = data["Sales"].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the custom Random Forest model
rf = RandomForestRegressorCustom(n_estimators=10, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)

# Print sample predictions
print("Predictions:", y_pred[:10])
