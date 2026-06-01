"""Tests for the AG Table plugin configuration."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


class FakeAgGrid:
    """Minimal stand-in for dash_ag_grid.AgGrid."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeDataFrame:
    """Small DataFrame stand-in for AGTable.update_properties()."""

    columns = ["Name", "Score"]

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def to_dict(self, orient):
        assert orient == "records"
        return self._rows


def load_ag_table_module(monkeypatch):
    """Load ag_table.py with lightweight stubs for optional UI dependencies."""
    pandas_module = ModuleType("pandas")
    pandas_module.DataFrame = FakeDataFrame
    monkeypatch.setitem(sys.modules, "pandas", pandas_module)

    dash_ag_grid_module = ModuleType("dash_ag_grid")
    dash_ag_grid_module.AgGrid = FakeAgGrid
    monkeypatch.setitem(sys.modules, "dash_ag_grid", dash_ag_grid_module)

    plugin_interface_module = ModuleType("workbench.web_interface.components.plugin_interface")
    plugin_interface_module.PluginInterface = object
    plugin_interface_module.PluginPage = SimpleNamespace(NONE="none")
    plugin_interface_module.PluginInputType = SimpleNamespace(DATAFRAME="dataframe")
    monkeypatch.setitem(sys.modules, "workbench.web_interface.components.plugin_interface", plugin_interface_module)

    symbols_module = ModuleType("workbench.utils.symbols")
    symbols_module.tag_symbols = lambda value: value
    monkeypatch.setitem(sys.modules, "workbench.utils.symbols", symbols_module)

    module_path = (
        Path(__file__).parents[2] / "src" / "workbench" / "web_interface" / "components" / "plugins" / "ag_table.py"
    )
    spec = importlib.util.spec_from_file_location("ag_table_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ag_table_shows_floating_filters_by_default(monkeypatch):
    module = load_ag_table_module(monkeypatch)
    table = module.AGTable()

    component = table.create_component("table")

    options = component.kwargs["dashGridOptions"]
    assert options["defaultColDef"]["filter"] is True
    assert options["defaultColDef"]["floatingFilter"] is True
    assert options["floatingFiltersHeight"] == table.floating_filter_height


def test_ag_table_height_includes_floating_filter_row(monkeypatch):
    module = load_ag_table_module(monkeypatch)
    table = module.AGTable()
    table.create_component("table", max_height=500)

    _, _, style = table.update_properties(FakeDataFrame([{"Name": "a", "Score": 1}, {"Name": "b", "Score": 2}]))

    expected_height = table.header_height + table.floating_filter_height + table.row_height * 2 + 2
    assert style["height"] == f"{expected_height}px"
