"""Minimal reproduction of cleanlab Datalab regression bug."""

import sys
from cleanlab import Datalab
import cleanlab
import datasets

print(f"Python: {sys.version}")
print(f"cleanlab: {cleanlab.__version__}")
print(f"datasets: {datasets.__version__}")

print("\n" + "=" * 60)
print("Attempting to create Datalab with task='regression'...")
print("=" * 60)

data = {"X": [0, 1, 2], "y": [0.1, 0.5, 0.9]}
lab = Datalab(data, label_name="y", task="regression")
