"""Plot Utilities for Workbench"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# For approximating beeswarm effect
def beeswarm_offsets(values, point_size=0.05, precision=2, max_offset=0.3):
    """
    Generate optimal beeswarm offsets with a maximum limit.

    Args:
        values: Array of positions to be adjusted
        point_size: Diameter of each point
        precision: Rounding precision for grouping
        max_offset: Maximum allowed offset in either direction

    Returns:
        Array of offsets for each point
    """
    values = np.asarray(values)
    rounded = np.round(values, precision)
    offsets = np.zeros_like(values, dtype=float)

    # Sort indices by original values
    sorted_idx = np.argsort(values)

    for val in np.unique(rounded):
        # Get indices belonging to this group
        group_idx = sorted_idx[np.isin(sorted_idx, np.where(rounded == val)[0])]

        if len(group_idx) > 1:
            # Track occupied positions for collision detection
            occupied = []

            for idx in group_idx:
                # Find best position with no collision
                offset = 0
                direction = 1
                step = 0

                while True:
                    # Check if current offset position is free
                    collision = any(abs(offset - pos) < point_size for pos in occupied)

                    if not collision or abs(offset) >= max_offset:
                        # Accept position if no collision or max offset reached
                        if abs(offset) > max_offset:
                            # Clamp to maximum
                            offset = max_offset * (1 if offset > 0 else -1)
                        break

                    # Switch sides with increasing distance
                    step += 0.25
                    direction *= -1
                    offset = direction * step * point_size

                offsets[idx] = offset
                occupied.append(offset)

    return offsets


def generate_heteroskedastic_data(n=1000, noise_factor=1.0, x_range=(0, 10)):
    """
    Generate data with heteroskedastic errors (increasing variance).

    Parameters:
    -----------
    n : int
        Number of data points
    noise_factor : float
        Controls the amount of noise (higher = more noise)
    x_range : tuple
        Range of x values (min, max)

    Returns:
    --------
    pd.DataFrame with columns 'x' and 'y'
    """
    # Generate x values
    x = np.linspace(x_range[0], x_range[1], n)

    # Base function (linear)
    y_base = 2 + 0.5 * x

    # Generate heteroskedastic noise (increasing with x)
    noise = np.random.normal(0, noise_factor * (0.1 * x), n)

    # Final y with increasing noise
    y = y_base + noise

    return pd.DataFrame({"id": range(n), "x": x, "y": y})


def prediction_intervals(df, figure, x_col):
    """
    Add prediction interval bands to a plotly figure based on percentile columns.
    Parameters:
    -----------
    df : pandas.DataFrame
        A Dataframe containing the data with percentile columns
    figure : plotly.graph_objects.Figure
        Plotly figure to add the prediction intervals to
    x_col : str
        Name of the x-axis column
    Returns:
    --------
    plotly.graph_objects.Figure
        Updated figure with prediction intervals
    """

    # Does the dataframe have the quantiles computed already?
    required_cols = ["q_025", "q_25", "q_75", "q_975"]
    if not all(col in df.columns for col in required_cols):
        # Check for a prediction_std column and compute quantiles if needed
        if "prediction_std" not in df.columns:
            return figure  # No quantiles to plot

        # Calculate quantiles based on standard deviation
        df["q_025"] = df["prediction"] - 1.96 * df["prediction_std"]
        df["q_975"] = df["prediction"] + 1.96 * df["prediction_std"]
        df["q_25"] = df["prediction"] - 0.674 * df["prediction_std"]
        df["q_75"] = df["prediction"] + 0.674 * df["prediction_std"]

    # Sort dataframe by x_col for connected lines
    sorted_df = df.sort_values(by=x_col)
    # Add outer band (q_025 to q_975) - more transparent
    figure.add_trace(
        go.Scatter(
            x=sorted_df[x_col],
            y=sorted_df["q_025"],
            mode="lines",
            line=dict(width=1, color="rgba(99, 110, 250, 0.25)"),
            name="2.5 Percentile",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sorted_df[x_col],
            y=sorted_df["q_975"],
            mode="lines",
            line=dict(width=1, color="rgba(99, 110, 250, 0.25)"),
            name="97.5 Percentile",
            hoverinfo="skip",
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.2)",
        )
    )
    # Add inner band (q_25 to q_75) - less transparent
    figure.add_trace(
        go.Scatter(
            x=sorted_df[x_col],
            y=sorted_df["q_25"],
            mode="lines",
            line=dict(width=1, color="rgba(99, 250, 110, 0.25)"),
            name="25 Percentile",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sorted_df[x_col],
            y=sorted_df["q_75"],
            mode="lines",
            line=dict(width=1, color="rgba(99, 250, 100, 0.25)"),
            name="75 Percentile",
            hoverinfo="skip",
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(99, 250, 110, 0.2)",
        )
    )
    return figure


if __name__ == "__main__":
    """Exercise the Plot Utilities"""
    import plotly.express as px

    # Test the beeswarm offsets function
    values = np.array([1, 1.01, 1.02, 2.02, 2.04, 3, 4, 5, 5.01, 5.02, 5.03, 5.04, 5.05, 6])
    jittered_values = beeswarm_offsets(values, precision=1)
    print("Jittered values:", jittered_values)

    # Generate noisy data
    df = generate_heteroskedastic_data()

    # Quick visualization with Plotly
    fig = px.scatter(
        df,
        x="x",
        y="y",
        opacity=0.7,
        title="Heteroskedastic Data for Quantile Regression",
        labels={"x": "X Variable", "y": "Y Variable"},
    )

    # Add a simple linear regression line to visualize the mean trend
    fig.add_scatter(x=df["x"], y=2 + 0.5 * df["x"], mode="lines", name="True Mean", line=dict(color="red"))

    fig.show()
