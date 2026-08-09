"""Two-way deep links between the browser URL and an artifact table's selected row.

Artifact pages carry their selection in the query string as ``?name=<artifact>``, so a page
can be linked to directly and a link copied out of the address bar reproduces what's on
screen. :func:`sync_selection` registers both directions.

Both run clientside -- the row data and the URL are already in the browser -- so neither
direction costs a server round-trip.
"""

from dash import clientside_callback, Input, Output, State

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
