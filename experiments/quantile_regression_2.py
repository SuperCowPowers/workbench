import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# SageWorks Imports
from sageworks.utils.test_data_generator import TestDataGenerator

# Get synthetic data
data_generator = TestDataGenerator()
df = data_generator.confidence_data()

feature_list = ["feature_1"]
feature = feature_list[0]
target = "target"
X_train = df[feature_list]
y_train = df[target]


# Get real data
if True:
    df = data_generator.aqsol_data()
    feature_list = data_generator.aqsol_features()
    # feature = "mollogp"
    # feature = "tpsa"
    # feature = "numhacceptors"
    # feature_list = ["mollogp", "tpsa", "numhacceptors"]
    feature = "mollogp"  # feature_list[0]
    target = data_generator.aqsol_target()
    X_train = df[feature_list]
    X_train = df[[feature]]
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


# Calculate confidence based on the quantile predictions
def domain_specific_confidence(quantile_models, X, predictions):
    lower_05 = quantile_models[0.05].predict(X)
    lower_25 = quantile_models[0.25].predict(X)
    quant_50 = quantile_models[0.50].predict(X)
    upper_75 = quantile_models[0.75].predict(X)
    upper_95 = quantile_models[0.95].predict(X)

    # Target sensitivity
    target_sensitivity = 1  # 0.25

    # Domain specific logic for calculating confidence
    # If the interval with is greater than target_sensitivity with have 0 confidence
    # anything below that is a linear scale from 0 to 1
    confidence_interval = upper_95 - lower_05
    q_conf = 1 - (confidence_interval / target_sensitivity)
    print(f"q_conf: {np.min(q_conf):.2f} {np.max(q_conf):.2f}")
    q_conf_clip = np.clip(q_conf, 0, 1)

    # Now lets look at the IQR distance for each observation
    epsilon_iqr = target_sensitivity * 0.5
    iqr = np.maximum(epsilon_iqr, np.abs(upper_75 - lower_25))
    iqr_distance = np.abs(predictions - quant_50) / iqr
    print(f"iqr_distance: {np.min(iqr_distance):.2f} {np.max(iqr_distance):.2f}")
    iqr_conf = 1 - iqr_distance
    print(f"iqr_conf: {np.min(iqr_conf):.2f} {np.max(iqr_conf):.2f}")
    iqr_conf_clip = np.clip(iqr_conf, 0, 1)

    # Now combine the two confidence values
    confidence = (q_conf_clip + iqr_conf_clip) / 2
    return confidence, confidence_interval, iqr_distance


def domain_specific_confidence_norm(quantile_models, X, predictions):
    lower_05 = quantile_models[0.05].predict(X)
    lower_25 = quantile_models[0.25].predict(X)
    quant_50 = quantile_models[0.50].predict(X)
    upper_75 = quantile_models[0.75].predict(X)
    upper_95 = quantile_models[0.95].predict(X)

    # Target sensitivity
    # target_sensitivity = 1  # 0.25

    # Domain specific logic for calculating confidence
    # If the interval with is greater than target_sensitivity with have 0 confidence
    # anything below that is a linear scale from 0 to 1
    conf_interval = upper_95 - lower_05
    print(f"confidence_interval: {np.min(conf_interval):.2f} {np.max(conf_interval):.2f}")

    # Normalize the confidence_interval between 0 and 1
    norm_conf_interval = (conf_interval - np.min(conf_interval)) / (np.max(conf_interval) - np.min(conf_interval))

    # Confidence is just 1 - conf_interval
    q_conf = 1 - norm_conf_interval

    # Now lets look at the IQR distance for each observation
    epsilon_iqr = 0.5
    iqr = np.maximum(epsilon_iqr, np.abs(upper_75 - lower_25))
    iqr_distance = np.abs(predictions - quant_50) / iqr
    print(f"iqr_distance: {np.min(iqr_distance):.2f} {np.max(iqr_distance):.2f}")

    # Normalize the iqr_distance between 0 and 1
    norm_iqr_distance = (iqr_distance - np.min(iqr_distance)) / (np.max(iqr_distance) - np.min(iqr_distance))

    # Confidence is just 1 - iqr_distance
    iqr_conf = 1 - norm_iqr_distance

    # Now combine the two confidence values
    confidence = (q_conf + iqr_conf) / 2
    return confidence, conf_interval, iqr_distance


def domain_specific_confidence_2(quantile_models, X, predictions):
    lower_05 = quantile_models[0.05].predict(X)
    lower_25 = quantile_models[0.25].predict(X)
    median_pred = quantile_models[0.50].predict(X)
    upper_75 = quantile_models[0.75].predict(X)
    upper_95 = quantile_models[0.95].predict(X)

    # Target sensitivity
    target_sensitivity = 0.25

    # Ensure IQR is positive
    epsilon = 1e-6
    iqr = np.maximum(epsilon, np.abs(upper_75 - lower_25))

    # Calculate confidence scores
    iqr_conf = np.clip(1 - (iqr / target_sensitivity), 0, 1)
    confidence = iqr_conf
    return confidence, iqr_conf, median_pred


# Fit the KNN model
def fit_knn_model(X, y, n_neighbors=5):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn


# Confidence method using the KNN model
def knn_confidence(knn, X_new, predictions=None, tolerance=0.1):
    # Get the neighbors' target values
    neighbors = knn.kneighbors(X_new, return_distance=False)
    neighbors_targets = np.array([knn._y[indices] for indices in neighbors])

    # Calculate the variance of the neighbors' target values
    variance = np.var(neighbors_targets, axis=1)

    # Confidence score can be inversely related to the variance
    confidence_scores = np.clip(1 - (variance / np.max(variance)), 0, 1)

    if predictions is not None:
        # Get KNN predictions for the new data
        knn_predictions = knn.predict(X_new)

        # Calculate the difference between provided predictions and KNN predictions
        prediction_diff = np.abs(predictions - knn_predictions)

        # Adjust confidence scores based on the prediction difference
        adjusted_confidence = np.clip(1 - (prediction_diff / tolerance), 0, 1)
        confidence_scores = np.minimum(confidence_scores, adjusted_confidence)

    return confidence_scores


# Train models
prediction_model = train_prediction_model(X_train, y_train)
quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
quantile_models = train_quantile_models(X_train, y_train, quantiles)

# Predictions
rmse_predictions = prediction_model.predict(X_train)

# Calculate confidence for the array of predictions
conf, conf_interval, iqr_distance = domain_specific_confidence_norm(quantile_models, X_train, rmse_predictions)

# Now we're going to use the KNN model to calculate confidence
knn = fit_knn_model(X_train, y_train)
knn_conf = knn_confidence(knn, X_train, rmse_predictions)

# Total confidence is the product of the two confidence scores
total_confidence = (conf + knn_conf) / 2

# Compute model metrics for RMSE
rmse = np.sqrt(np.mean((y_train - rmse_predictions) ** 2))
print(f"RMSE: {rmse} support: {len(X_train)}")

# Domain Specific Confidence Threshold
thres = 0.8
conf_for_thres = conf

# Now filter the data based on confidence and give RMSE for the filtered data
rmse_predictions_filtered = rmse_predictions[conf_for_thres > thres]
y_train_filtered = y_train[conf_for_thres > thres]
rmse_filtered = np.sqrt(np.mean((y_train_filtered - rmse_predictions_filtered) ** 2))
print(f"RMSE Filtered: {rmse_filtered} support: {len(rmse_predictions_filtered)}")


# Plot the results
plt.figure(figsize=(10, 6))
actual_values = y_train
actual_values = df["mollogp"]
sc = plt.scatter(actual_values, y_train, c=conf, cmap="coolwarm", label="Data", alpha=0.5)
plt.colorbar(sc, label="Confidence")

# Sort x_values and the corresponding y-values for each quantile
"""
sorted_indices = np.argsort(actual_values)
sorted_actual_values = actual_values[sorted_indices]
for q in quantiles:
    sorted_y_values = quantile_models[q].predict(X_train)[sorted_indices]
    plt.plot(sorted_actual_values, sorted_y_values, label=f"Quantile {int(q * 100):02}")


# Plot the RMSE predictions
plt.plot(sorted_actual_values, rmse_predictions[sorted_indices], label="RMSE Prediction", color="black", linestyle="--")
"""
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Quantile Regression and Confidence with XGBoost")
plt.legend()
plt.show()
