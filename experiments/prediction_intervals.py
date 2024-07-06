import numpy as np
import matplotlib.pyplot as plt

# Example data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_true = 3 * x + np.random.normal(0, 1, 100)
y_pred = 3 * x

# Prediction intervals
y_pred_lower = y_pred - 1.96
y_pred_upper = y_pred + 1.96

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, "o", label="True Values")
plt.plot(x, y_pred, "-", label="Predicted Values")
plt.fill_between(
    x,
    y_pred_lower,
    y_pred_upper,
    color="gray",
    alpha=0.2,
    label="95% Prediction Interval",
)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Prediction Intervals")
plt.show()
