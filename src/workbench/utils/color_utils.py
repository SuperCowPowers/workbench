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

    print("Color Utilities tests pass.")
