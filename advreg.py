import pandas as pd
import statsmodels.api as sm

# Load the dataset
file_path = "Advertising.csv"  # Update with the correct path if needed
df = pd.read_csv(file_path)

# Define independent (X) and dependent (y) variables
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Compute Residual Standard Error (RSE), R-squared, and F-statistic
rse = model.mse_resid ** 0.5
r_squared = model.rsquared
f_statistic = model.fvalue

# Print results
print("Regression Summary:")
print(model.summary())
print(f"\nResidual Standard Error (RSE): {rse:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"F-statistic: {f_statistic:.4f}")
