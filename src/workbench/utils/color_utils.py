"""Color Utilities for Workbench"""

import re


def is_dark(color: str) -> bool:
    """
    Determines if an rgba(...) color is dark based on the average of its RGB values.

    Args:
        color (str): The color in rgba(...) format.

    Returns:
        bool: True if the color is dark, False otherwise.
    """
    match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color)
    if not match:
        raise ValueError(f"Invalid color format: {color}")

    r, g, b = map(int, match.groups())
    return (r + g + b) / 3 < 128


def color_to_rgba(color: str) -> str:
    """
    Converts a color string in various formats (HEX, RGB, RGBA) to an RGBA string.

    Args:
        color (str): The input color in hex, RGB, or RGBA format.

    Returns:
        str: The color in RGBA format (e.g., "rgba(70, 145, 220, 0.5)").
    """
    if color.startswith("rgba"):
        # Already in RGBA format
        return color

    if color.startswith("rgb"):
        # Convert RGB to RGBA by adding alpha if missing
        if "a" not in color:
            color = color.replace("rgb", "rgba").rstrip(")") + ", 1.0)"
        return color

    if color.startswith("#"):
        # Convert HEX to RGBA
        hex_color = color.lstrip("#")
        if len(hex_color) == 6:  # #RRGGBB
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return f"rgba({r}, {g}, {b}, 1.0)"
        elif len(hex_color) == 8:  # #RRGGBBAA
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            a = int(hex_color[6:8], 16) / 255
            return f"rgba({r}, {g}, {b}, {a:.1f})"

    raise ValueError(f"Unsupported color format: {color}")


def rgba_to_tuple(rgba: str) -> tuple[float, float, float, float]:
    """
    Converts an rgba(...) string to a normalized tuple of (R, G, B, A), where R, G, B are in the range [0, 1].

    Args:
        rgba (str): The RGBA color string (e.g., "rgba(255, 0, 0, 0.5)").

    Returns:
        tuple[float, float, float, float]: A normalized tuple of (R, G, B, A).
    """
    components = rgba.strip("rgba() ").split(",")
    r, g, b = (int(components[i]) / 255 for i in range(3))  # Normalize RGB values
    a = float(components[3])  # Alpha is already normalized
    return r, g, b, a


def adjust_towards_gray(r: float, g: float, b: float, factor: float = 0.5) -> tuple[float, float, float]:
    """Shift RGB values towards gray."""
    gray = 0.5  # Middle gray in normalized scale (0–1)
    r = r + (gray - r) * factor
    g = g + (gray - g) * factor
    b = b + (gray - b) * factor
    return r, g, b


# Colorscale interpolation
def weights_to_colors(weights: list[float], colorscale: list, muted: bool = True) -> list[str]:
    """
    Map a list of weights to colors using Plotly's sample_colorscale function and return rgba strings.

    Args:
        weights (list[float]): A list of weights (0 to 1) to map to colors.
        colorscale (list): A Plotly-style colorscale.
        muted (bool): Whether to mute the colors (shift towards gray).

    Returns:
        list[str]: A list of interpolated colors in rgba format.
    """
    from plotly.colors import sample_colorscale

    # Handle normalization of weights
    min_weight, max_weight = min(weights), max(weights)
    if min_weight == max_weight:
        # If all weights are the same, normalize to a single value (e.g., 0)
        normalized_weights = [0.0] * len(weights)
    else:
        # Normalize weights to the range [0, 1]
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]

    # Sample colors for normalized weights
    sampled_colors = sample_colorscale(colorscale, normalized_weights, colortype="rgba")

    result_colors = []
    for rgba in sampled_colors:
        # Handle RGB or RGBA output
        if len(rgba) == 3:  # RGB
            r, g, b = rgba
            a = 1.0
        elif len(rgba) == 4:  # RGBA
            r, g, b, a = rgba

        # Optionally shift colors towards gray
        if muted:
            r, g, b = adjust_towards_gray(r, g, b)

        # Convert to rgba string
        result_colors.append(f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})")

    return result_colors


def remove_middle_colors(colorscale, min_threshold=0.3, max_threshold=0.7):
    """Removes colors between min_threshold and max_threshold from a colorscale
    and adds transparent versions of the closest colors at the thresholds.

    Args:
        colorscale (list): List of (position, color) tuples where position is between 0 and 1.
        min_threshold (float, optional): Lower bound for the range to make transparent. Defaults to 0.35.
        max_threshold (float, optional): Upper bound for the range to make transparent. Defaults to 0.65.

    Returns:
        list: Modified colorscale with middle colors removed and transparent colors added at thresholds.
    """
    new_colorscale = []
    has_min_threshold = False
    has_max_threshold = False

    # Sort the colorscale by position to ensure we process in order
    sorted_colorscale = sorted(colorscale, key=lambda x: x[0])

    # Helper function to make a color transparent
    def make_transparent(color_str):
        # Handle different color formats
        if color_str.startswith("rgb"):
            if color_str.startswith("rgba"):
                # Replace the alpha value
                return color_str.rsplit(",", 1)[0] + ",0)"
            else:
                # Convert rgb to rgba with 0 alpha
                return color_str.replace("rgb", "rgba").rstrip(")") + ",0)"
        elif color_str.startswith("#"):
            # Convert hex to rgba
            hex_color = color_str.lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join([c * 2 for c in hex_color])
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"rgba({r},{g},{b},0)"
        # Return as-is if format is not recognized
        return color_str

    for i, (pos, color) in enumerate(sorted_colorscale):
        # Keep colors outside the range
        if pos < min_threshold or pos > max_threshold:
            new_colorscale.append((pos, color))

        # Check if we need to add a color at min_threshold
        if not has_min_threshold and pos >= min_threshold:
            # Find the previous point for interpolation
            if i > 0 and sorted_colorscale[i - 1][0] < min_threshold:
                prev_pos, prev_color = sorted_colorscale[i - 1]
                new_colorscale.append((min_threshold, make_transparent(prev_color)))
                has_min_threshold = True

        # Check if we need to add a color at max_threshold
        if not has_max_threshold and pos > max_threshold:
            # Find the previous point for interpolation
            if i > 0 and sorted_colorscale[i - 1][0] <= max_threshold:
                transparent_color = make_transparent(sorted_colorscale[i - 1][1])
                new_colorscale.append((max_threshold, transparent_color))
                has_max_threshold = True

    # If we've gone through all points and haven't added thresholds yet
    if not has_min_threshold and len(sorted_colorscale) > 0:
        # Find closest color below min_threshold
        closest_below = None
        min_dist = float("inf")
        for p, c in sorted_colorscale:
            if p < min_threshold and min_threshold - p < min_dist:
                closest_below = c
                min_dist = min_threshold - p

        if closest_below:
            new_colorscale.append((min_threshold, make_transparent(closest_below)))
        elif sorted_colorscale:
            new_colorscale.append((min_threshold, make_transparent(sorted_colorscale[0][1])))

    if not has_max_threshold and len(sorted_colorscale) > 0:
        # Find closest color above max_threshold
        closest_above = None
        min_dist = float("inf")
        for p, c in sorted_colorscale:
            if p > max_threshold and p - max_threshold < min_dist:
                closest_above = c
                min_dist = p - max_threshold

        if closest_above:
            new_colorscale.append((max_threshold, make_transparent(closest_above)))
        elif sorted_colorscale:
            new_colorscale.append((max_threshold, make_transparent(sorted_colorscale[-1][1])))

    # Re-sort the final colorscale
    return sorted(new_colorscale, key=lambda x: x[0])


if __name__ == "__main__":
    """Exercise the Color Utilities"""

    # Test the is_dark function
    assert is_dark("rgba(0, 0, 0, 1.0)") is True
    assert is_dark("rgba(255, 255, 255, 1.0)") is False

    # Test the color_to_rgba function
    assert color_to_rgba("#000000") == "rgba(0, 0, 0, 1.0)"
    assert color_to_rgba("#000000FF") == "rgba(0, 0, 0, 1.0)"
    assert color_to_rgba("#FFFFFF") == "rgba(255, 255, 255, 1.0)"
    assert color_to_rgba("#FFFFFF00") == "rgba(255, 255, 255, 0.0)"
    assert color_to_rgba("rgb(0, 0, 0)") == "rgba(0, 0, 0, 1.0)"
    assert color_to_rgba("rgb(255, 255, 255)") == "rgba(255, 255, 255, 1.0)"
    assert color_to_rgba("rgba(0, 0, 0, 0.5)") == "rgba(0, 0, 0, 0.5)"
    assert color_to_rgba("rgba(255, 255, 255, 0.5)") == "rgba(255, 255, 255, 0.5)"

    # Test the weights_to_colors function
    colorscale = [
        [0.0, "rgba(255, 0, 0, 0.5)"],
        [0.5, "rgb(255, 255, 0)"],
        [1.0, "rgb(0, 255, 0)"],
    ]
    print(weights_to_colors([0.0, 0.2, 0.4, 0.8, 1.0], colorscale))
    print("Color Utilities tests pass.")

    # Test the remove_middle_colors function
    new_colorscale = remove_middle_colors(colorscale)
    print("Modified colorscale:", new_colorscale)
