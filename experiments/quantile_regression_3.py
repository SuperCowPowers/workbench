import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Generate synthetic data with even spacing from -10 to 5 and sparse spacing from 5 to 10
x_even = np.linspace(-10, 5, 1600)  # Evenly spaced from -10 to 5
x_sparse = 5 + (np.linspace(0, 1, 200) ** 2) * 5  # Increasingly sparse from 5 to 10
x = np.concatenate([x_even, x_sparse])

# Ensure no values are exactly zero or negative in the input to the log function
epsilon = 1e-6  # Small value to avoid log(0)
x_adjusted = np.where(x >= 0, x + 1 + epsilon, -x + 1 + epsilon)

# Generate non-linear 'S' shape y values
np.random.seed(42)
# y = np.where(x >= 0, np.log10(x_adjusted), -np.log10(x_adjusted)) + np.random.normal(scale=0.02 * np.abs(x), size=x.shape)
y = np.where(x >= 0, np.log(x_adjusted) / np.log(100), -np.log(x_adjusted) / np.log(100)) + np.random.normal(
    scale=0.02 * np.abs(x), size=x.shape
)

# Add pairs coincident points in X to test IQR functionality
for i in range(3):
    x_coincident = np.array([-0.5, -0.5, 0, 0, 0.5, 0.5])

    # Increasing deltas for the y values
    y_delta = 0.1 + 0.05 * i
    y_offsets = [-0.1, 0, 0.1]

    # Now create pairs of y values (-delta + offset and +delta + offset) for each x value
    y_coincident = np.concatenate([[-y_delta + y_offset, y_delta + y_offset] for y_offset in y_offsets])

    # Add the coincident points to the data`
    x = np.concatenate([x, x_coincident])
    y = np.concatenate([y, y_coincident])

"""
x_coincident = np.array([-0.5, -0.5, 0, 0, 0.5, 0.5])
y_coincident = np.array([-.4, .1, -.25, .25, -.1, .4])
x = np.concatenate([x, x_coincident])
y = np.concatenate([y, y_coincident])
"""

# Reshape the data
X_train = x.reshape(-1, 1)
y_train = y

# Sort X for plotting (also sorts y)
sort_idx = np.argsort(X_train[:, 0])
X_train = X_train[sort_idx]
y_train = y_train[sort_idx]


# Function to train and predict using the prediction model
def train_prediction_model(X, y):
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)
    return model


# Function to train quantile models and store predictions
def train_quantile_models(X, y, quantiles, n_estimators=100):
    quantile_models = {}
    for quantile in quantiles:
        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=quantile, n_estimators=n_estimators, max_depth=1
        )
        model.fit(X, y)
        quantile_models[quantile] = model
    return quantile_models


# Calculate confidence based on the quantile predictions
def calculate_confidence(y, quantile_models, X):
    lower_05 = quantile_models[0.05].predict(X)
    lower_25 = quantile_models[0.25].predict(X)
    quant_50 = quantile_models[0.50].predict(X)
    upper_75 = quantile_models[0.75].predict(X)
    upper_95 = quantile_models[0.95].predict(X)

    # Target sensitivity
    target_sensitivity = 0.25

    # Domain specific logic for calculating confidence
    # If the interval with is greater than target_sensitivity with have 0 confidence
    # anything below that is a linear scale from 0 to 1
    confidence_interval = upper_95 - lower_05
    q_conf = np.clip((1 - confidence_interval) / (target_sensitivity * 4.0), 0, 1)

    # Now lets look at the IQR distance for each observation
    epsilon_iqr = target_sensitivity * 0.5
    iqr = np.maximum(epsilon_iqr, np.abs(upper_75 - lower_25))
    iqr_distance = np.abs(y - quant_50) / iqr
    iqr_conf = np.clip(1 - iqr_distance, 0, 1)

    # Now combine the two confidence values
    confidence = (q_conf + iqr_conf) / 2
    return confidence


# Train models
prediction_model = train_prediction_model(X_train, y_train)
quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
quantile_models = train_quantile_models(X_train, y_train, quantiles)

# Predictions
rmse_predictions = prediction_model.predict(X_train)

# Calculate confidence for the array of predictions
confidence_values = calculate_confidence(y_train, quantile_models, X_train)

# Plot the results
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_train, y_train, c=confidence_values, cmap="coolwarm_r", label="Data", alpha=0.5)
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
