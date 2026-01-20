"""Clientside JavaScript callbacks for Dash components.

These functions return JavaScript code strings for use with Dash's clientside_callback.
Using clientside callbacks avoids server round-trips for simple UI interactions.
"""


def circle_overlay_callback(circle_data_uri: str) -> str:
    """Returns JS function for circle overlay on scatter plot hover.

    Args:
        circle_data_uri: Base64-encoded SVG data URI for the circle overlay

    Returns:
        JavaScript function string for use with clientside_callback
    """
    return f"""
    function(hoverData) {{
        if (!hoverData) {{
            return [false, window.dash_clientside.no_update, window.dash_clientside.no_update];
        }}
        var bbox = hoverData.points[0].bbox;
        var centerX = (bbox.x0 + bbox.x1) / 2;
        var centerY = (bbox.y0 + bbox.y1) / 2;
        var adjustedBbox = {{
            x0: centerX - 50,
            x1: centerX + 50,
            y0: centerY - 162,
            y1: centerY - 62
        }};
        var imgElement = {{
            type: 'Img',
            namespace: 'dash_html_components',
            props: {{
                src: '{circle_data_uri}',
                style: {{width: '100px', height: '100px'}}
            }}
        }};
        return [true, adjustedBbox, [imgElement]];
    }}
    """
