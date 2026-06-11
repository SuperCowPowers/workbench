"""Shared pytest configuration for the workbench test suite"""

import os
import pytest

import plotly.graph_objects as go


@pytest.fixture(autouse=True)
def headless_plotly(monkeypatch):
    """No-op figure.show() on CI runners (no browser there). GitHub Actions
    sets CI=true; locally show() still pops a browser for plot eyeballing."""
    if os.getenv("CI"):
        monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)
