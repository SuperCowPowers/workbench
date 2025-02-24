import matplotlib.pyplot as plt
import numpy as np


def plot_grouped_bar_chart():
    # Data
    row_sizes = ["10", "100", "500", "1000", "10000"]
    taut = [1.0616, 2.0423, 9.3037, 15.0423, 120.3992]
    md = [0.5407, 1.1638, 4.1056, 6.8408, 59.3434]
    model = [0.4811, 0.7123, 0.9522, 1.6606, 10.0377]
    pipeline = [1.4450, 2.8939, 12.2653, 21.3893, 190.1198]
    pipeline_fast = [1.2368, 1.8528, 8.9519, 12.8159, 110.5273]

    # Split data for two plots
    subsets = {
        "Small Row Sizes": (["10", "100", "500"], taut[:3], md[:3], model[:3], pipeline[:3], pipeline_fast[:3]),
        "Large Row Sizes": (["1000", "10000"], taut[3:], md[3:], model[3:], pipeline[3:], pipeline_fast[3:]),
    }

    for title, (rows, taut_vals, md_vals, model_vals, pipeline_vals, pipeline_fast_vals) in subsets.items():
        x = np.arange(len(rows))
        width = 0.25  # Width of each bar

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        # Individual stacked bar
        ax.bar(x - width, taut_vals, width, label="Taut", color="lightblue")
        ax.bar(x - width, md_vals, width, bottom=taut_vals, label="MD", color="lightgreen")
        ax.bar(
            x - width,
            model_vals,
            width,
            bottom=np.array(taut_vals) + np.array(md_vals),
            label="Model",
            color="lightcoral",
        )

        # Pipeline and Pipeline Fast bars
        ax.bar(x, pipeline_vals, width, label="Pipeline", color="blue", alpha=0.7)
        ax.bar(x + width, pipeline_fast_vals, width, label="Pipeline Fast", color="green", alpha=0.7)

        # Labels and legend
        ax.set_xticks(x)
        ax.set_xticklabels(rows)
        ax.set_xlabel("Number of Rows")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f"Grouped Bar Chart: {title}")
        ax.legend()

        plt.show()


plot_grouped_bar_chart()
