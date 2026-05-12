"""Utility helpers for web_interface components."""

import base64


def circle_overlay_data_uri(marker_size: int = 15) -> str:
    """Build a base64-encoded SVG data URI for a hover circle overlay.

    Used by ``circle_overlay_callback`` to outline the hovered marker on
    scatter-like plots. The circle radius scales with the Plotly marker size
    so the overlay sits just outside the marker edge.

    Args:
        marker_size (int): Plotly marker size (diameter, in pixels) of the
            scatter points the overlay will outline. Default 15.

    Returns:
        str: A ``data:image/svg+xml;base64,...`` string usable as an ``<img>`` src.
    """
    # At marker_size=15 the original used r=10 — gives a couple of pixels of
    # clearance between the marker edge and the inner edge of the stroke.
    radius = marker_size / 2 + 2.5

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
        'style="overflow: visible;">'
        f'<circle cx="50" cy="50" r="{radius}" '
        'stroke="rgba(255, 255, 255, 1)" stroke-width="3" fill="none" />'
        "</svg>"
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"
