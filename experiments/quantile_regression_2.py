import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Generate synthetic data
np.random.seed(42)
x = np.linspace(-10, 10, 200)
y = x + np.random.normal(scale=0.2 * np.abs(x), size=x.shape)
X_train = x.reshape(-1, 1)
y_train = y


# Function to train quantile model and predict
def train_quantile_model(X, y, quantile, n_estimators):
    model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=quantile, n_estimators=100, max_depth=1)
    model.fit(X, y)
    return model.predict(X)


# Function to train RMSE model and predict
def train_rmse_model(X, y, n_estimators):
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)
    return model.predict(X)


# Number of estimators
n_estimators = 10

# Compute quantiles
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
predictions = {}

for q in quantiles:
    predictions[q] = train_quantile_model(X_train, y_train, q, n_estimators)

# Compute RMSE model predictions
rmse_predictions = train_rmse_model(X_train, y_train, n_estimators)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, label="Data", alpha=0.5)
colors = ["blue", "green", "red", "purple", "orange"]
for q, color in zip(quantiles, colors):
    plt.plot(X_train, predictions[q], label=f"Quantile {int(q*100)}", color=color)
plt.plot(X_train, rmse_predictions, label="RMSE Prediction", color="black", linestyle="--")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Quantile Regression and RMSE Prediction with XGBoost")
plt.legend()
plt.show()
