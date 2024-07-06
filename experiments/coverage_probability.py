import numpy as np
import matplotlib.pyplot as plt

# Example coverage probability data
nominal_confidence_levels = np.linspace(0.5, 1.0, 10)
coverage_probabilities = nominal_confidence_levels + np.random.normal(0, 0.05, 10)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(
    nominal_confidence_levels,
    coverage_probabilities,
    "o-",
    label="Coverage Probability",
)
plt.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray")
plt.xlabel("Nominal Confidence Level")
plt.ylabel("Coverage Probability")
plt.legend()
plt.title("Coverage Probability Plot")
plt.show()
