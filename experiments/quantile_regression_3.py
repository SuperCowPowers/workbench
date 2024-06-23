import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Generate synthetic data with even spacing from -10 to 5 and sparse spacing from 5 to 10
x_even = np.linspace(-10, 5, 800)  # Evenly spaced from -10 to 5
x_sparse = 5 + (np.linspace(0, 1, 50) ** 2) * 5  # Increasingly sparse from 5 to 10
x = np.concatenate([x_even, x_sparse])

# Ensure no values are exactly zero or negative in the input to the log function
epsilon = 1e-6  # Small value to avoid log(0)
x_adjusted = np.where(x >= 0, x + 1 + epsilon, -x + 1 + epsilon)

# Generate non-linear 'S' shape y values
y = np.where(x >= 0, np.log(x_adjusted), -np.log(x_adjusted)) + np.random.normal(scale=0.05 * np.abs(x), size=x.shape)

# Reshape the data
X_train = x.reshape(-1, 1)
y_train = y


# Function to train and predict using the prediction model
def train_prediction_model(X, y):
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)
    return model


# Function to train quantile models and store predictions
def train_quantile_models(X, y, quantiles, n_estimators=200):
    quantile_models = {}
    for quantile in quantiles:
        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=quantile, n_estimators=n_estimators, max_depth=1
        )
        model.fit(X, y)
        quantile_models[quantile] = model
    return quantile_models


# Calculate confidence based on the quantile predictions
def calculate_confidence(preds, quantile_models, X):
    lower_05 = quantile_models[0.05].predict(X)
    lower_25 = quantile_models[0.25].predict(X)
    quant_50 = quantile_models[0.50].predict(X)
    upper_75 = quantile_models[0.75].predict(X)
    upper_95 = quantile_models[0.95].predict(X)

    # Domain specific logic for calculating confidence
    # If the interval with is greater than 1 with have 0 confidence
    # anything below that is a linear scale from 0 to 1
    interval_width = upper_95 - lower_05
    confidence = np.clip((1 - interval_width) / 1, 0, 1)
    return confidence

    """
    confidence = np.zeros_like(preds)
    for i, pred in enumerate(preds):
        if pred < lower_25[i]:
            confidence[i] = (pred - lower_05[i]) / (lower_25[i] - lower_05[i])
        elif pred > upper_75[i]:
            confidence[i] = (upper_95[i] - pred) / (upper_95[i] - upper_75[i])
        else:
            confidence[i] = 1.0  # High confidence if the prediction is between the 25th and 75th percentiles

    return confidence
    """


# Train models
prediction_model = train_prediction_model(X_train, y_train)
quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
quantile_models = train_quantile_models(X_train, y_train, quantiles)

# Predictions
rmse_predictions = prediction_model.predict(X_train)

# Calculate confidence for the array of predictions
confidence_values = calculate_confidence(rmse_predictions, quantile_models, X_train)

# Plot the results
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_train, y_train, c=confidence_values, cmap="coolwarm", label="Data", alpha=0.5)
# sc = plt.scatter(X_train, y_train, cmap='coolwarm', label='Data', alpha=0.5)
plt.colorbar(sc, label="Confidence")
for q in quantiles:
    plt.plot(X_train, quantile_models[q].predict(X_train), label=f"Quantile {int(q * 100)}")
plt.plot(X_train, rmse_predictions, label="RMSE Prediction", color="black", linestyle="--")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Quantile Regression and Confidence with XGBoost")
plt.legend()
plt.show()
