"""Clientside JavaScript callbacks for Dash components.

These functions return JavaScript code strings for use with Dash's clientside_callback.
Using clientside callbacks avoids server round-trips for simple UI interactions.
"""


def circle_overlay_callback(circle_data_uri: str) -> str:
    """Return JS that shows a white circle overlay on scatter plot hover.

    Args:
        circle_data_uri (str): Base64-encoded SVG data URI for the circle image.

    Returns:
        str: JavaScript function string for use with clientside_callback.
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
            x0: centerX - 50, x1: centerX + 50,
            y0: centerY - 162, y1: centerY - 62
        }};
        var imgElement = {{
            type: 'Img',
            namespace: 'dash_html_components',
            props: {{ src: '{circle_data_uri}', style: {{width: '100px', height: '100px'}} }}
        }};
        return [true, adjustedBbox, [imgElement]];
    }}
    """


def external_highlight_callback(graph_id: str) -> str:
    """Return JS that triggers a real Plotly hover via synthetic mouse event.

    Finds the point by mol_id in plotDiv.data, locates the corresponding SVG
    ``<path>`` element, reads its bounding rect, and dispatches a ``mousemove``
    on Plotly's drag layer. This triggers Plotly's own hover machinery, which
    computes the correct bbox internally — no manual pixel math needed.

    Args:
        graph_id (str): DOM id of the dcc.Graph wrapper element.

    Returns:
        str: JavaScript function string (dummy output — hover callbacks do the real work).
    """
    return f"""
    function(storeData) {{
        if (!storeData || !storeData.mol_id) {{ return window.dash_clientside.no_update; }}

        var wrapper = document.getElementById('{graph_id}');
        if (!wrapper) {{ return window.dash_clientside.no_update; }}
        var plotDiv = wrapper.querySelector('.js-plotly-plot');
        if (!plotDiv || !plotDiv.data) {{ return window.dash_clientside.no_update; }}

        /* Find trace index and point index for this mol_id */
        var data = plotDiv.data;
        var traceIdx = -1, pointIdx = -1;
        for (var t = 0; t < data.length; t++) {{
            var cd = data[t].customdata;
            if (!cd) continue;
            for (var p = 0; p < cd.length; p++) {{
                if (cd[p] && cd[p][1] === storeData.mol_id) {{
                    traceIdx = t;
                    pointIdx = p;
                    break;
                }}
            }}
            if (traceIdx >= 0) break;
        }}
        if (traceIdx < 0) {{ return window.dash_clientside.no_update; }}

        /* Find the SVG <path> for this point.
           Plotly renders each go.Scatter trace as a <g class="trace scatter">
           inside .scatterlayer, with individual points as <path> children
           of the .points sub-group. */
        var traces = plotDiv.querySelectorAll('.scatterlayer .trace');
        if (traceIdx >= traces.length) {{ return window.dash_clientside.no_update; }}
        var points = traces[traceIdx].querySelectorAll('.points path');
        if (pointIdx >= points.length) {{ return window.dash_clientside.no_update; }}
        var svgPoint = points[pointIdx];

        /* Get the center of the SVG element in viewport coordinates */
        var rect = svgPoint.getBoundingClientRect();
        var cx = (rect.left + rect.right) / 2;
        var cy = (rect.top + rect.bottom) / 2;

        /* Dispatch mousemove on Plotly's drag overlay to trigger real hover */
        var dragLayer = plotDiv.querySelector('.nsewdrag');
        if (dragLayer) {{
            dragLayer.dispatchEvent(new MouseEvent('mousemove', {{
                clientX: cx, clientY: cy, bubbles: true
            }}));
        }}

        return window.dash_clientside.no_update;
    }}
    """
