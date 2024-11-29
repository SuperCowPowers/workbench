"""A set of color map helper functions"""


def hex_to_rgb(hex_color: str, alpha: float = 0.2) -> str:
    """Internal: Convert a hex color to rgb
    Args:
        hex_color: The hex color to convert
        alpha: The alpha value to use (default: 0.2)
    Returns:
        str: The rgb color
    """
    hex_color = hex_color.lstrip("#")
    rgb = f"rgba({int(hex_color[:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:], 16)}, {alpha})"
    return rgb


def color_map_add_alpha(color_map: list, alpha: float = 0.2) -> list:
    """Internal: Add alpha to the given color map
    Args:
        color_map: The color map to add alpha to
        alpha: The alpha value to use (default: 0.2)
    Returns:
        list: The color map with alpha added
    """
    return [hex_to_rgb(color_map[i], alpha) for i in range(len(color_map))]
