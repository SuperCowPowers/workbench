"""A Markdown Component for details/information about DataSources (and FeatureSets)"""

from dash import dcc

# SageWorks Imports
from sageworks.web_components.component_interface import ComponentInterface


class DataDetailsMarkdown(ComponentInterface):
    """Data Details Markdown Component"""

    def create_component(self, component_id: str) -> dcc.Markdown:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Markdown: The Dash Markdown Component
        """
        waiting_markdown = "*Waiting for data...*"
        return dcc.Markdown(id=component_id, children=waiting_markdown, dangerously_allow_html=True)

    def generate_markdown(self, data_details: dict) -> str:
        """Create the Markdown for the details/information about the DataSource or the FeatureSet
        Args:
            data_details (dict): A dictionary of information about the artifact
        Returns:
            str: A Markdown string
        """

        markdown_template = """
        **Rows:** <<num_rows>>
        <br>**Columns:** <<num_columns>>
        <br>**Created/Mod:** <<created>> / <<modified>>
        <br>**Tags:** <<sageworks_tags>>
        <br>**Input:** <<input>>

        #### Numeric Columns
        <<numeric_column_details>>

        #### Non-Numeric Columns
        <<string_column_details>>
        """

        expanding_list = """
        <details>
            <summary><<column_info>></summary>
            <ul>
            <<bullet_list>>
            </ul>
        </details>
        """

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
                    bullet_list += f"<li>{q}: {value:.3f}</li>"
                bullet_list += f"<li>Unique: {column_info['unique']}</li>"

                # Add correlations if they exist
                if column_info.get("correlations"):
                    corr_title = """<span class="lightgreen"><b>Correlated Columns</b></span>"""
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
            display_fs_type = shorten_map.get(column_info["fs_dtype"], "???")
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
        html_template = """<b><<name>></b> <span class="lightblue">(<<full_type>>)</span>:"""

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
            if column_info["dtype"] in float_types:
                html_template += f""" {min:.2f} → {max:.2f}&nbsp;&nbsp;&nbsp;&nbsp;"""
            else:
                html_template += f""" {int(min)} → {int(max)}&nbsp;&nbsp;&nbsp;&nbsp;"""
            if column_info["unique"] == 2 and min == 0 and max == 1:
                html_template += """ <span class="lightgreen"> Binary</span>"""
            elif column_info["num_zeros"] > 0:
                html_template += """ <span class="lightorange"> Zero: <<num_zeros>></span>"""

        # Non-numeric columns get the number of unique values
        else:
            html_template += """ Unique: <<unique>> """

        # Do we have any nulls in this column?
        if column_info["nulls"] > 0:
            html_template += """ <span class="lightred">Null: <<nulls>></span>"""

        # Replace the column name
        html_template = html_template.replace("<<name>>", column_name)

        # Construct the full type
        column_info = self._construct_full_type(column_info)

        # Loop through all the details and replace in the template
        for key, value in column_info.items():
            html_template = html_template.replace(f"<<{key}>>", str(value))

        return html_template


if __name__ == "__main__":
    # This class takes in model details and generates a Confusion Matrix
    import dash
    from dash import dcc
    from dash import html
    from sageworks.api.data_source import DataSource

    # Create the class and get the AWS FeatureSet details
    ds = DataSource("wine_data")
    data_details = ds.details()

    # Instantiate the DataDetailsMarkdown class
    ddm = DataDetailsMarkdown()
    component = ddm.create_component("data_details_markdown")

    # Generate the markdown
    markdown = ddm.generate_markdown(data_details)

    # Initialize Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([component])
    component.children = markdown

    if __name__ == "__main__":
        app.run(debug=True)
