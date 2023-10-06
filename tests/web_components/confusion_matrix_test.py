import os
import sys
import pandas as pd

# Get the path from the environment variable
sys.path.append(os.getenv('SAGEWORKS_PLUGINS')) if os.getenv('SAGEWORKS_PLUGINS') else None
from confusion_matrix import ConfusionMatrix


if __name__ == "__main__":
    # Instantiate the ConfusionMatrix class
    cm = ConfusionMatrix()

    # Create some test data
    test_data = {
        'low': [0.6, 0.2, 0.2],
        'med': [0.1, 0.7, 0.2],
        'high': [0.1, 0.1, 0.8]
    }
    test_df = pd.DataFrame(test_data, index=['low', 'med', 'high'])

    # Generate the figure
    fig = cm.generate_component_figure(test_df)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()