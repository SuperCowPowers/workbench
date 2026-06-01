"""Shared loading placeholders for dashboard callbacks."""

import plotly.graph_objects as go

WAITING_MARKDOWN = "*Waiting for data...*"
WAITING_HEADER = "Loading..."
WAITING_TABLE_TEXT = "Waiting for data..."
WAITING_FIGURE_TEXT = "Waiting for Data..."


def waiting_figure(text_message: str = WAITING_FIGURE_TEXT) -> go.Figure:
    """Return the same style of text-only figure used by unloaded graph components."""
    figure = go.Figure()
    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=text_message,
        bgcolor="rgba(0,0,0,0)",
        showarrow=False,
        font=dict(size=24, color="#9999cc"),
    )
    figure.update_layout(
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def waiting_table() -> tuple[list[dict], list[dict]]:
    """Return a one-row table placeholder for dependent row-selection tables."""
    return [{"headerName": "Status", "field": "status"}], [{"status": WAITING_TABLE_TEXT}]
