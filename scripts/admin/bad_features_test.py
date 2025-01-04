import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Original dataset
X_original = pd.DataFrame(np.random.rand(1000, 10))  # 10 original features
y = (X_original.sum(axis=1) > 5).astype(int)  # Simple binary target

# Add random noise features
X_noise = X_original.copy()
X_noise["random_feature"] = np.random.rand(1000)

# Add adversarial feature (anti-correlated with target)
X_noise["adversarial_feature"] = -y + np.random.normal(0, 0.1, size=1000)
X_noise["random_noise"] = np.random.rand(1000)

# Train and compare models
model = XGBClassifier()

original_score = cross_val_score(model, X_original, y, cv=5, scoring="accuracy").mean()
noise_score = cross_val_score(model, X_noise, y, cv=5, scoring="accuracy").mean()

print(f"Original CV Accuracy: {original_score:.4f}")
print(f"CV Accuracy with Adversarial Features: {noise_score:.4f}")

model.fit(X_noise, y)
print(model.feature_importances_)
