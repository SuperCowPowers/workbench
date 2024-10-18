import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder

# Generate a small test dataframe with class A, B, and C probabilities
data = {
    "true_class": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "C"],
    "pred_class": ["C", "C", "C", "C", "C", "C", "C", "C", "C", "C"],
    "A_proba": [0.3, 0.3, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3, 0.2],
    "B_proba": [0.1, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
    "C_proba": [0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6],
}

# Create the DataFrame
df = pd.DataFrame(data)

# Extract class labels and probabilities
class_labels = ["A", "B", "C"]
y_true = df["true_class"]

# One-hot encode the true labels
encoder = OneHotEncoder(categories=[class_labels], sparse_output=False)
y_true_encoded = encoder.fit_transform(df[["true_class"]])

# Extract predicted probabilities for each class
y_score = df[["A_proba", "B_proba", "C_proba"]]

# Calculate ROC AUC using "all-in-one" approach
roc_auc_all_in_one = roc_auc_score(y_true_encoded, y_score, multi_class="ovr", average="macro")

# Calculate ROC AUC for each class individually (per-class approach)
roc_auc_per_class = []
for i, label in enumerate(class_labels):
    y_true_class = y_true_encoded[:, i]  # True labels for this class
    y_score_class = y_score.iloc[:, i]  # Predicted probabilities for this class

    # Calculate ROC AUC for this class
    auc = roc_auc_score(y_true_class, y_score_class)
    roc_auc_per_class.append(auc)

# Sanity checks
#
# 1: Ensure that the highest probability corresponds to the predicted class (XGBoost logic)
# 2: Ensure that all the probabilities sum to 1
prediction_col = y_score.idxmax(axis=1).str.replace("_proba", "")
assert df["pred_class"].equals(prediction_col)
assert y_score.sum(axis=1).eq(1).all()

# Get the precision, recall, and f-score for each class
precision, recall, fscore, support = precision_recall_fscore_support(
    y_true, df["pred_class"], labels=class_labels, zero_division=0
)

# Format the results for each class
for i, label in enumerate(class_labels):
    print(f"\nClass: {label}")
    print(f"  Precision: {precision[i]:.2f}")
    print(f"  Recall: {recall[i]:.2f}")
    print(f"  F1-Score: {fscore[i]:.2f}")
    print(f"  ROC AUC (per class): { roc_auc_per_class[i]}")
    print(f"  Averaged ROC AUC (OVR): {roc_auc_all_in_one}")
