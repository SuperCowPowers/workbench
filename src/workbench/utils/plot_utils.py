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


def prediction_intervals(df, figure, x_col, smoothing=0):
    """
    Add prediction interval bands to a plotly figure based on percentile columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the data with percentile columns
    figure : plotly.graph_objects.Figure
        Plotly figure to add the prediction intervals to
    x_col : str
        Name of the x-axis column
    smoothing : int, default=0
        Size of the rolling window for smoothing
        0 = no smoothing, larger values create smoother bands

    Returns:
    --------
    plotly.graph_objects.Figure
        Updated figure with prediction intervals
    """
    required_cols = ["q_05", "q_25", "q_75", "q_95"]
    if all(col in df.columns for col in required_cols):
        # Sort dataframe by x_col for connected lines
        sorted_df = df.sort_values(by=x_col)

        # Apply smoothing if requested
        if smoothing > 0:
            # Use odd window size for centered smoothing
            smoothing = smoothing if smoothing % 2 == 1 else smoothing + 1

            # Apply appropriate aggregation for each percentile
            # Lower percentiles use min to avoid underestimating lower bounds
            sorted_df["q_05"] = sorted_df["q_05"].rolling(window=smoothing, center=True, min_periods=1).min()
            sorted_df["q_05"] = sorted_df["q_05"].ewm(span=smoothing, min_periods=1).mean()
            sorted_df["q_25"] = sorted_df["q_25"].rolling(window=smoothing, center=True, min_periods=1).min()
            sorted_df["q_25"] = sorted_df["q_25"].ewm(span=smoothing, min_periods=1).mean()

            # Upper percentiles use max to avoid overestimating upper bounds
            sorted_df["q_75"] = sorted_df["q_75"].rolling(window=smoothing, center=True, min_periods=1).max()
            sorted_df["q_75"] = sorted_df["q_75"].ewm(span=smoothing, min_periods=1).mean()
            sorted_df["q_95"] = sorted_df["q_95"].rolling(window=smoothing, center=True, min_periods=1).max()
            sorted_df["q_95"] = sorted_df["q_95"].ewm(span=smoothing, min_periods=1).mean()

        # Add outer band (q_05 to q_95) - more transparent
        figure.add_trace(
            go.Scatter(
                x=sorted_df[x_col],
                y=sorted_df["q_05"],
                mode="lines",
                line=dict(width=1, color="rgba(99, 110, 250, 0.5)", dash="dash"),
                name="5th Percentile",
                hoverinfo="none",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=sorted_df[x_col],
                y=sorted_df["q_95"],
                mode="lines",
                line=dict(width=1, color="rgba(99, 110, 250, 0.5)", dash="dash"),
                name="95th Percentile",
                hoverinfo="none",
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
                line=dict(width=1, color="rgba(99, 110, 250, 0.5)", dash="dash"),
                name="25th Percentile",
                hoverinfo="none",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=sorted_df[x_col],
                y=sorted_df["q_75"],
                mode="lines",
                line=dict(width=1, color="rgba(99, 110, 250, 0.5)", dash="dash"),
                name="75th Percentile",
                hoverinfo="none",
                fill="tonexty",
                fillcolor="rgba(99, 110, 250, 0.2)",
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

    # Generate noisey data
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
