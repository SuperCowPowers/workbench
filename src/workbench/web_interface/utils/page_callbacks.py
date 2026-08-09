"""Callbacks that wire an artifact page into standard dashboard behavior.

Two pieces, both registered from a page's setup:

* :func:`sync_selection` -- two-way deep links between the URL's ``?name=<artifact>`` and the
  table's selected row, so a page can be linked to directly and a link copied out of the
  address bar reproduces what's on screen. Both directions run clientside; the row data and
  the URL are already in the browser, so neither costs a server round-trip.
* :func:`register_theme_callback` -- re-render plugin figures on a theme change. CSS follows
  the theme on its own, but a Plotly figure is baked at render time, so pages that skip this
  keep their plots on the old theme until a full reload.
"""

from dash import callback, clientside_callback, Input, Output, State

# The dashboard's theme store (defined in app.py). A clientside callback writes the selected
# theme name to it after setting the wb_theme cookie, so any request that follows already
# resolves to the new theme and the re-render only has to rebuild.
#
# Plugins re-render through register_theme_callback() below. Plain ComponentInterface
# components hold no data between callbacks, so a page re-renders those by taking this store
# as an extra Input on the callback that already builds the figure.
THEME_STORE_ID = "workbench-theme-store"

# Select the row named by ?name=, or the first row when the query string is absent. The
# store latches once a selection lands, so periodic table refreshes fall straight through.
# A name that matches nothing selects nothing and leaves the store unlatched, so a later
# refresh that brings the row in still gets a chance to select it.
_SELECT_FROM_URL_JS = """
function (rowData, pageAlreadyLoaded) {
    var noUpdate = window.dash_clientside.no_update;
    if (pageAlreadyLoaded || !rowData || !rowData.length) {
        return [noUpdate, noUpdate];
    }
    var name = new URLSearchParams(window.location.search).get("name");
    if (!name) {
        return [[rowData[0]], true];
    }
    var row = rowData.find(function (r) { return r.name === name; });
    return row ? [[row], true] : [noUpdate, noUpdate];
}
"""

# Rewrite the query string in place. Writing to dcc.Location instead would be a navigation:
# with refresh="callback-nav" that remounts the page on every selection. Setting "name" on
# the existing params rather than replacing the query string keeps any other param intact.
_WRITE_TO_URL_JS = """
function (selectedRows) {
    if (selectedRows && selectedRows[0] && selectedRows[0].name) {
        var params = new URLSearchParams(window.location.search);
        params.set("name", selectedRows[0].name);
        var search = "?" + params.toString();
        if (search !== window.location.search) {
            window.history.replaceState(null, "", search);
        }
    }
    return window.dash_clientside.no_update;
}
"""


def plugin_outputs(plugins: list, allow_duplicate: bool = False) -> list:
    """Every plugin property, flattened into callback Outputs.

    Args:
        plugins (list): The PluginInterface instances to aggregate.
        allow_duplicate (bool): Set when another callback already writes these properties.

    Returns:
        list: An Output per (component_id, property) across all the plugins.
    """
    return [Output(cid, prop, allow_duplicate=allow_duplicate) for p in plugins for cid, prop in p.properties]


def sync_selection(table_id: str, loaded_store_id: str):
    """Register the callbacks that keep ``?name=`` and the table selection in step.

    Args:
        table_id (str): The AGTable's component id, e.g. "models_table"
        loaded_store_id (str): A dcc.Store id holding a bool, e.g. "models_page_loaded"
    """
    clientside_callback(
        _SELECT_FROM_URL_JS,
        Output(table_id, "selectedRows"),
        Output(loaded_store_id, "data"),
        Input(table_id, "rowData"),
        State(loaded_store_id, "data"),
        prevent_initial_call=True,
    )

    # The output is a formality -- the JS always returns no_update and does its work through
    # the History API -- so it reuses the store above rather than requiring a dummy element.
    clientside_callback(
        _WRITE_TO_URL_JS,
        Output(loaded_store_id, "data", allow_duplicate=True),
        Input(table_id, "selectedRows"),
        prevent_initial_call=True,
    )


def register_theme_callback(plugins: list):
    """Re-render every plugin's figures when the theme changes.

    Args:
        plugins (list): The PluginInterface instances on the page.
    """

    @callback(
        plugin_outputs(plugins, allow_duplicate=True),
        Input(THEME_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def on_theme_change(theme):
        all_props = []
        for plugin in plugins:
            all_props.extend(plugin.set_theme(theme))
        return all_props
