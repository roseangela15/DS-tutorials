import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Convert labels from {0,1} to {-1,1} for hinge loss
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

# Load the Advertising Dataset
if __name__ == "__main__":
    # Read dataset
    df = pd.read_csv("C:/Users/Rose Angela Vinod/Desktop/DS TUTORIALS/Tutorial4/advertising.csv")
    
    # Define features and target
    X = df[['TV', 'Radio', 'Newspaper']].values
    median_sales = df['Sales'].median()
    y = (df['Sales'] > median_sales).astype(int).values  # Convert Sales into binary labels
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train SVM
    model = SVM()
    model.fit(X_train, y_train)
    
    # Predict on test data
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
