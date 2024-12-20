"""Main: The main Workbench Web Interface to view, interact, and manage Workbench Artifacts"""

from dash import register_page

# Workbench Imports
from workbench.web_interface.page_views.main_page import MainPage
from workbench.web_interface.components.plugins.ag_table import AGTable

# Local Imports
from .layout import main_layout
from . import callbacks

register_page(
    __name__,
    path="/",
    name="Workbench",
)

# Create a table object for each Workbench Artifact Type
tables = dict()
table_names = ["data_sources", "feature_sets", "models", "endpoints"]
for table_name in table_names:
    tables[table_name] = AGTable()

# Create a table container for each Workbench Artifact Type
table_containers = dict()
table_name = "data_sources"
table_containers[table_name] = tables[table_name].create_component(
    f"main_{table_name}",
    header_color="rgb(120, 70, 70)",
    max_height=200,
)
table_name = "feature_sets"
table_containers[table_name] = tables[table_name].create_component(
    f"main_{table_name}",
    header_color="rgb(110, 110, 70)",
    max_height=200,
)
table_name = "models"
table_containers[table_name] = tables[table_name].create_component(
    f"main_{table_name}",
    header_color="rgb(60, 100, 60)",
    max_height=500,
)
table_name = "endpoints"
table_containers[table_name] = tables[table_name].create_component(
    f"main_{table_name}",
    header_color="rgb(100, 60, 100)",
    max_height=500,
)

# Set up our layout (Dash looks for a var called layout)
layout = main_layout(**table_containers)

# Grab a view that gives us a summary of all the artifacts currently in Workbench
main_page_view = MainPage()

# Set up the callbacks for tables on the main page
callbacks.last_updated()
callbacks.plugin_page_info()
callbacks.tables_refresh(main_page_view, tables)

# Set up our subpage navigation
callbacks.navigate_to_subpage()
