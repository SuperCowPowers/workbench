"""Shared pytest configuration for the workbench test suite"""

import os

# On CI runners there's no browser, so figure.show() calls in tests (a local
# dev nicety for eyeballing plots) become a no-op. GitHub Actions sets CI=true.
if os.getenv("CI"):
    import plotly.graph_objects as go

    go.Figure.show = lambda self, *args, **kwargs: None
