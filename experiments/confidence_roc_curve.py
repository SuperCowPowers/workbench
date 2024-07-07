import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Generate more accurate fake data with correlation between confidence and accuracy
np.random.seed(42)
n_samples = 1000
actual_values = np.random.uniform(1.5, 4.5, n_samples)

# Split actual_values into three parts
split_index1 = int(n_samples / 3)
split_index2 = int(2 * n_samples / 3)

actual_values_high = actual_values[:split_index1]
actual_values_medium = actual_values[split_index1:split_index2]
actual_values_low = actual_values[split_index2:]

# High confidence -> low error
high_confidence_predictions = actual_values_high + np.random.normal(0, 0.1, split_index1)
high_confidence_scores = np.random.uniform(0.8, 1.0, split_index1)

# Medium confidence -> moderate error
medium_confidence_predictions = actual_values_medium + np.random.normal(0, 0.3, split_index2 - split_index1)
medium_confidence_scores = np.random.uniform(0.5, 0.8, split_index2 - split_index1)

# Low confidence -> high error
low_confidence_predictions = actual_values_low + np.random.normal(0, 0.5, n_samples - split_index2)
low_confidence_scores = np.random.uniform(0.0, 0.5, n_samples - split_index2)

# Combine all predictions and confidence scores
predictions = np.concatenate(
    [
        high_confidence_predictions,
        medium_confidence_predictions,
        low_confidence_predictions,
    ]
)
confidence_scores = np.concatenate([high_confidence_scores, medium_confidence_scores, low_confidence_scores])

# Define accuracy threshold (e.g., absolute error < 0.5)
accuracy_threshold = 0.5
absolute_errors = np.abs(predictions - np.concatenate([actual_values_high, actual_values_medium, actual_values_low]))
accurate = absolute_errors < accuracy_threshold

# Categorize confidence scores into high, medium, and low
confidence_bins = np.percentile(confidence_scores, [33, 66])
low_confidence = confidence_scores <= confidence_bins[0]
medium_confidence = (confidence_scores > confidence_bins[0]) & (confidence_scores <= confidence_bins[1])
high_confidence = confidence_scores > confidence_bins[1]


# Compute ROC curve and AUC for each category
def compute_roc(accurate, confidence_category):
    fpr, tpr, _ = roc_curve(accurate, confidence_category)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


fpr_low, tpr_low, roc_auc_low = compute_roc(accurate, low_confidence)
fpr_medium, tpr_medium, roc_auc_medium = compute_roc(accurate, medium_confidence)
fpr_high, tpr_high, roc_auc_high = compute_roc(accurate, high_confidence)

# Plot ROC curves
plt.figure()
plt.plot(
    fpr_low,
    tpr_low,
    color="red",
    lw=2,
    label=f"Low Confidence ROC (area = {roc_auc_low:.2f})",
)
plt.plot(
    fpr_medium,
    tpr_medium,
    color="orange",
    lw=2,
    label=f"Medium Confidence ROC (area = {roc_auc_medium:.2f})",
)
plt.plot(
    fpr_high,
    tpr_high,
    color="green",
    lw=2,
    label=f"High Confidence ROC (area = {roc_auc_high:.2f})",
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Different Confidence Levels")
plt.legend(loc="lower right")
plt.show()
