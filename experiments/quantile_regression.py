import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Workbench Imports
from workbench.utils.test_data_generator import TestDataGenerator

# Get synthetic data
data_generator = TestDataGenerator()
df = data_generator.confidence_data()

feature_list = ["feature_1"]
feature = feature_list[0]
target = "target"
X_train = df[feature_list]
y_train = df[target]


# Get real data
if False:
    df = data_generator.aqsol_data()
    feature_list = data_generator.aqsol_features()
    # feature = "mollogp"
    # feature = "tpsa"
    # feature = "numhacceptors"
    # feature_list = ["mollogp", "tpsa", "numhacceptors"]
    feature = "mollogp"  # feature_list[0]
    target = data_generator.aqsol_target()
    X_train = df[feature_list]
    y_train = df[target]


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
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            n_estimators=n_estimators,
            max_depth=1,
        )
        model.fit(X, y)
        quantile_models[quantile] = model
    return quantile_models


def normalize(data):
    data = np.array(data)
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data


# Calculate confidence based on the quantile predictions
def calculate_confidence(quantile_models, X, y):
    lower_05 = quantile_models[0.05].predict(X)
    lower_25 = quantile_models[0.25].predict(X)
    quant_50 = quantile_models[0.50].predict(X)
    upper_75 = quantile_models[0.75].predict(X)
    upper_95 = quantile_models[0.95].predict(X)

    # Domain specific logic for calculating confidence
    # If the interval with is greater than target_sensitivity with have 0 confidence
    # anything below that is a linear scale from 0 to 1
    confidence_interval = upper_95 - lower_05
    q_conf = 1 - confidence_interval
    print(f"q_conf: {np.min(q_conf):.2f} {np.max(q_conf):.2f}")

    # Normalize the confidence to be between 0 and 1
    q_conf = normalize(q_conf)
    print(f"q_conf: {np.min(q_conf):.2f} {np.max(q_conf):.2f}")

    # Now lets look at the IQR distance for each observation
    target_sensitivity = 1.0
    epsilon_iqr = target_sensitivity * 0.5
    iqr = np.maximum(epsilon_iqr, np.abs(upper_75 - lower_25))
    iqr_distance = np.abs(y - quant_50) / iqr
    iqr_conf = 1 - iqr_distance
    print(f"iqr_conf: {np.min(iqr_conf):.2f} {np.max(iqr_conf):.2f}")

    # Normalize the confidence to be between 0 and 1
    iqr_conf = normalize(iqr_conf)
    print(f"q_conf: {np.min(iqr_conf):.2f} {np.max(iqr_conf):.2f}")

    # Now combine the two confidence values
    confidence = (q_conf + iqr_conf) / 2
    return confidence


# Train the Prediction Model
prediction_model = train_prediction_model(X_train, y_train)
rmse_predictions = prediction_model.predict(X_train)
residuals = y_train - rmse_predictions

# Train the Quantiles Models (on the residuals)
quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
quantile_models = train_quantile_models(X_train, y_train, quantiles)

# Calculate confidence for the array of predictions
confidence_values = calculate_confidence(quantile_models, X_train, y_train)

# Compute model metrics for RMSE
rmse = np.sqrt(np.mean((y_train - rmse_predictions) ** 2))
print(f"RMSE: {rmse} support: {len(X_train)}")

# Confidence Threshold
conf_thres = 0.8

# Now filter the data based on confidence and give RMSE for the filtered data
rmse_predictions_filtered = rmse_predictions[confidence_values > conf_thres]
y_train_filtered = y_train[confidence_values > conf_thres]
rmse_filtered = np.sqrt(np.mean((y_train_filtered - rmse_predictions_filtered) ** 2))
print(f"RMSE Filtered: {rmse_filtered} support: {len(rmse_predictions_filtered)}")


# Plot the results
plt.figure(figsize=(10, 6))
x_values = df[feature]
# x_values = df["feature_1"]
# x_values = rmse_predictions

"""
# Sort both the x and y values (for plotting purposes)
sort_order = np.argsort(x_values)
x_values = x_values[sort_order]
y_train = y_train[sort_order]
"""
sc = plt.scatter(x_values, y_train, c=confidence_values, cmap="coolwarm", label="Data", alpha=0.5)
plt.colorbar(sc, label="Confidence")

# Sort x_values and the corresponding y-values for each quantile
sorted_indices = np.argsort(x_values)
sorted_x_values = x_values[sorted_indices]
for q in quantiles:
    sorted_y_values = quantile_models[q].predict(X_train)[sorted_indices]
    plt.plot(sorted_x_values, sorted_y_values, label=f"Quantile {int(q * 100):02}")

# Plot the RMSE predictions
plt.plot(
    sorted_x_values,
    rmse_predictions[sorted_indices],
    label="RMSE Prediction",
    color="black",
    linestyle="--",
)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Quantile Regression and Confidence with XGBoost")
plt.legend()
plt.show()
