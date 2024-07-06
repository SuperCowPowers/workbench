import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Example data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_true = 3 * x + np.random.normal(0, 1, 100)
y_pred = 3 * x

# Residuals
residuals = y_true - y_pred

# Plot Residuals vs. Fitted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line="45")
plt.title("Q-Q Plot")
plt.show()
