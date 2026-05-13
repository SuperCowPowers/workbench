"""Clientside JavaScript callbacks for Dash components.

These functions return JavaScript code strings for use with Dash's clientside_callback.
Using clientside callbacks avoids server round-trips for simple UI interactions.
"""


def hover_ring_overlay_callback(graph_id: str, overlay_name: str = "__hover_overlay__") -> str:
    """Return JS that moves a hidden overlay trace to the hovered point.

    Pairs with a single-point trace (named ``overlay_name``) appended to the figure.
    The callback uses ``Plotly.restyle`` to set the overlay's x/y to the hovered
    point's coordinates (or null on unhover). Since the ring is a real Plotly
    point, alignment is exact. Same pattern extends to line overlays (mode="lines")
    for things like dynamic node-to-node links.
    """
    return f"""
    function(hoverData) {{
        var plotDiv = (document.getElementById('{graph_id}') || {{}}).querySelector
            && document.getElementById('{graph_id}').querySelector('.js-plotly-plot');
        if (!plotDiv || !plotDiv.data || !window.Plotly) return window.dash_clientside.no_update;
        var idx = plotDiv.data.findIndex(function(t) {{ return t.name === '{overlay_name}'; }});
        if (idx < 0) return window.dash_clientside.no_update;
        var pt = (hoverData && hoverData.points && hoverData.points[0].curveNumber !== idx)
            ? hoverData.points[0] : null;
        var update = {{x: [[pt ? pt.x : null]], y: [[pt ? pt.y : null]]}};
        if (pt) {{
            // Plotly resolves per-point marker.size on the hover point when it's an array.
            // For scalar marker.size, fall back to reading it off the trace.
            var sz = plotDiv.data[pt.curveNumber].marker && plotDiv.data[pt.curveNumber].marker.size;
            var pointSize = typeof pt['marker.size'] === 'number' ? pt['marker.size'] : sz;
            if (typeof pointSize === 'number') update['marker.size'] = pointSize + 4;
        }}
        window.Plotly.restyle(plotDiv, update, [idx]);
        return window.dash_clientside.no_update;
    }}
    """


def external_highlight_callback(graph_id: str) -> str:
    """Return JS that triggers a real Plotly hover via synthetic mouse event.

    Finds the point by mol_id in plotDiv.data, locates the corresponding SVG
    ``<path>`` element, reads its bounding rect, and dispatches a ``mousemove``
    on Plotly's drag layer. This triggers Plotly's own hover machinery, which
    computes the correct bbox internally — no manual pixel math needed.

    Requires SVG-rendered scatter traces (``go.Scatter``, not ``go.Scattergl``)
    since it relies on per-point ``<path>`` DOM elements being queryable.

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
