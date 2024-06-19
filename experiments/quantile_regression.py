import xgboost as xgb
import numpy as np

# Generate synthetic training data
np.random.seed(42)
X_train = np.random.rand(100, 2) * 1000  # Features: [size, number of bedrooms]
y_train = X_train[:, 0] * 300 + X_train[:, 1] * 50000 + np.random.randn(100) * 50000  # House prices

# Train models for the 10th, 50th, and 90th percentiles
quantiles = [0.1, 0.50, 0.9]
models = {}
for q in quantiles:
    params = {"objective": "reg:quantileerror", "eval_metric": "mae", "quantile_alpha": q}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    models[q] = model

# Example test data
X_test = np.array([[2250, 4]])  # Features: [size, number of bedrooms]

# Predict the 10th, 50th, and 90th percentiles
preds = {}
for q in quantiles:
    preds[q] = models[q].predict(X_test)

# Output predictions
preds_10 = preds[0.1]  # Lower bound
preds_50 = preds[0.50]  # Median
preds_90 = preds[0.9]  # Upper bound

print(f"10th Percentile: ${preds_10[0]:,.0f}")
print(f"50th Percentile (Median): ${preds_50[0]:,.0f}")
print(f"90th Percentile: ${preds_90[0]:,.0f}")
