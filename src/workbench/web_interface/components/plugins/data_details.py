"""A Markdown Component for details/information about DataSources (and FeatureSets)"""

from typing import Union
import logging

# Dash Imports
from dash import html, dcc

# Workbench Imports
from workbench.api import DataSource, FeatureSet
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.markdown_utils import tags_to_markdown

# Get the Workbench logger
log = logging.getLogger("workbench")


class DataDetails(PluginInterface):
    """DataSource/FeatureSet Details Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATA_SOURCE

    def __init__(self):
        """Initialize the DataDetails plugin class"""
        self.component_id = None
        self.data_source = None

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for the Model Details
        """
        self.component_id = component_id
        container = html.Div(
            id=self.component_id,
            children=[
                html.H3(id=f"{self.component_id}-header", children="Loading..."),
                dcc.Markdown(id=f"{self.component_id}-details", dangerously_allow_html=True),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-details", "children"),
        ]

        # Return the container
        return container

    def update_properties(self, artifact: Union[DataSource, FeatureSet], **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            artifact (Union[DataSource, FeatureSet]): An instantiated DataSource or FeatureSet
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        log.important(f"Updating Plugin with Artifact: {artifact.uuid} and kwargs: {kwargs}")

        # Update the header and the details
        self.data_source = artifact.data_source if isinstance(artifact, FeatureSet) else artifact
        header = f"{self.data_source.uuid}"
        data_details_markdown = self.data_details_markdown(self.data_source.details())

        return [header, data_details_markdown]

    def data_details_markdown(self, data_details: dict) -> str:

        markdown_template = (
            "**Rows:** <<num_rows>>  \n"
            "**Columns:** <<num_columns>>  \n"
            "**Created/Mod:** <<created>> / <<modified>>  \n"
            "<<workbench_tags>>  \n"
            "**Input:** <<input>>  \n"
            "<br>  \n"
            "\n#### Numeric Columns  \n"
            "<<numeric_column_details>>  \n"
            "\n<div style='margin-top: 15px;'></div>  \n"
            "\n#### Non-Numeric Columns  \n"
            "<<string_column_details>>  \n"
        )

        expanding_list = (
            "<details>  \n"
            "    <summary><<column_info>></summary>  \n"
            "    <ul>  \n"
            "    <<bullet_list>>  \n"
            "    </ul>  \n"
            "</details>"
        )

        # Sanity Check for empty data
        if not data_details:
            return "No data source details found"

        # Loop through all the details and replace in the template
        for key, value in data_details.items():
            # Hack for dates
            if ".000Z" in str(value):
                try:
                    value = value.replace(".000Z", "").replace("T", " ")
                except AttributeError:
                    pass

            # Special case for tags
            if key == "workbench_tags":
                value = tags_to_markdown(value)
            markdown_template = markdown_template.replace(f"<<{key}>>", str(value))

        # Fill in numeric column details
        column_stats = data_details.get("column_stats", {})
        numeric_column_details = ""
        numeric_types = [
            "tinyint",
            "smallint",
            "int",
            "bigint",
            "float",
            "double",
            "decimal",
        ]
        for column_name, column_info in column_stats.items():
            if column_info["dtype"] in numeric_types:
                column_html = self._column_info_html(column_name, column_info)
                column_details = expanding_list.replace("<<column_info>>", column_html)

                # Populate the bullet list (descriptive_stats and unique)
                bullet_list = ""
                for q, value in column_info["descriptive_stats"].items():
                    if value is not None:
                        bullet_list += f"<li>{q}: {value:.3f}</li>"
                bullet_list += f"<li>Unique: {column_info['unique']}</li>"

                # Add correlations if they exist
                if column_info.get("correlations"):
                    corr_title = """<span class="green-text"><b>Correlated Columns</b></span>"""
                    corr_details = expanding_list.replace("<<column_info>>", corr_title)
                    corr_details = f"""<li class="no-bullet">{corr_details}</li>"""
                    corr_list = ""
                    for col, corr in column_info["correlations"].items():
                        corr_list += f"<li>{col}: {corr:.3f}</li>"
                    corr_details = corr_details.replace("<<bullet_list>>", corr_list)
                    bullet_list += corr_details

                # Add the bullet list to the column details
                column_details = column_details.replace("<<bullet_list>>", bullet_list)

                # Add the column details to the markdown
                numeric_column_details += column_details

        # Now actually replace the column details in the markdown
        markdown_template = markdown_template.replace("<<numeric_column_details>>", numeric_column_details)

        # For string columns create collapsible sections that show value counts
        string_column_details = ""
        for column_name, column_info in column_stats.items():
            # Skipping any columns that are numeric
            if column_info["dtype"] in numeric_types:
                continue

            # Skipping any columns that are dates/timestamps
            if column_info["dtype"] == "timestamp":
                continue

            # Create the column info
            column_html = self._column_info_html(column_name, column_info)
            column_details = expanding_list.replace("<<column_info>>", column_html)

            # Populate the bullet list (if we have value counts)
            if "value_counts" not in column_info:
                bullet_list = "<li>No Value Counts</li>"
            else:
                bullet_list = ""
                for value, count in column_info["value_counts"].items():
                    bullet_list += f"<li>{value}: {count}</li>"

            # Add the bullet list to the column details
            column_details = column_details.replace("<<bullet_list>>", bullet_list)

            # Add the column details to the markdown
            string_column_details += column_details

        # Now actually replace the column details in the markdown
        markdown = markdown_template.replace("<<string_column_details>>", string_column_details)
        return markdown

    @staticmethod
    def _construct_full_type(column_info: dict) -> dict:
        """Internal: Show the FeatureSet Types if they exist"""
        shorten_map = {
            "Integral": "I",
            "Fractional": "F",
            "String": "S",
            "Timestamp": "TS",
            "Boolean": "B",
        }
        if "fs_dtype" in column_info:
            display_fs_type = shorten_map.get(column_info["fs_dtype"], "V")
            column_info["full_type"] = f"{display_fs_type}: {column_info['dtype']}"
        else:
            column_info["full_type"] = column_info["dtype"]
        return column_info

    def _column_info_html(self, column_name, column_info: dict) -> str:
        """Internal: Create an HTML string for a column's information
        Args:
            column_name (str): The name of the column
            column_info (dict): A dictionary of column information
        Returns:
            str: An HTML string
        """

        # First part of the HTML template is the same for all columns
        html_template = """<b><<name>></b> <span class="blue-text">(<<full_type>>)</span>:"""

        # Add min, max, and number of zeros for numeric columns
        numeric_types = [
            "tinyint",
            "smallint",
            "int",
            "bigint",
            "float",
            "double",
            "decimal",
        ]
        float_types = ["float", "double", "decimal"]
        if column_info["dtype"] in numeric_types:
            # Just hardcode the min and max for now
            min = column_info["descriptive_stats"]["min"]
            max = column_info["descriptive_stats"]["max"]

            # Sanity Check
            if min is None or max is None:
                html_template += """ <span class="red-text">No Stats</span>"""

            # Floats get 2 decimal places
            elif column_info["dtype"] in float_types:
                html_template += f""" {min:.2f} → {max:.2f}&nbsp;&nbsp;&nbsp;&nbsp;"""

            # Integers get no decimal places
            else:
                html_template += f""" {int(min)} → {int(max)}&nbsp;&nbsp;&nbsp;&nbsp;"""
            if column_info["unique"] == 2 and min == 0 and max == 1:
                html_template += """ <span class="green-text"> Binary</span>"""
            elif column_info["num_zeros"] > 0:
                html_template += """ <span class="orange-text"> Zero: <<num_zeros>></span>"""

        # Non-numeric columns get the number of unique values
        else:
            html_template += """ Unique: <<unique>> """

        # Do we have any nulls in this column?
        if column_info["nulls"] > 0:
            html_template += """ <span class="red-text">Null: <<nulls>></span>"""

        # Replace the column name
        html_template = html_template.replace("<<name>>", column_name)

        # Construct the full type
        column_info = self._construct_full_type(column_info)

        # Loop through all the details and replace in the template
        for key, value in column_info.items():
            html_template = html_template.replace(f"<<{key}>>", str(value))

        return html_template


if __name__ == "__main__":
    # This class takes in model details and generates a details Markdown component
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(DataDetails).run()
