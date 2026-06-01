"""Tests for dashboard loading placeholders."""

import importlib.util
import sys
import types
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "src" / "workbench" / "web_interface" / "components" / "loading_state.py"


class _FakeFigure:
    def __init__(self):
        self.layout = types.SimpleNamespace(annotations=[])

    def add_annotation(self, **kwargs):
        self.layout.annotations.append(types.SimpleNamespace(**kwargs))

    def update_layout(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = types.SimpleNamespace(**value)
            setattr(self.layout, key, value)


fake_graph_objects = types.SimpleNamespace(Figure=_FakeFigure)
sys.modules.setdefault("plotly", types.SimpleNamespace(graph_objects=fake_graph_objects))
sys.modules.setdefault("plotly.graph_objects", fake_graph_objects)

SPEC = importlib.util.spec_from_file_location("loading_state", MODULE_PATH)
loading_state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loading_state)


def test_waiting_table_has_status_row():
    column_defs, row_data = loading_state.waiting_table()

    assert column_defs == [{"headerName": "Status", "field": "status"}]
    assert row_data == [{"status": "Waiting for data..."}]


def test_waiting_figure_has_centered_status_text():
    figure = loading_state.waiting_figure()

    assert figure.layout.annotations[0].text == "Waiting for Data..."
    assert figure.layout.annotations[0].showarrow is False
    assert figure.layout.xaxis.showticklabels is False
    assert loading_state.WAITING_MARKDOWN == "*Waiting for data...*"
