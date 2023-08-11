"""A Component for details/information about DataSources"""
from dash import dcc


def column_info_html(column_name, column_info: dict) -> str:
    """Create an HTML string for a column's information
    Args:
        column_name (str): The name of the column
        column_info (dict): A dictionary of column information
    Returns:
        str: An HTML string
    """

    # First part of the HTML template is the same for all columns
    html_template = """<b><<name>></b> <span class="lightblue">(<<dtype>>)</span>:"""

    # Add min, max, and number of zeros for numeric columns
    numeric_types = ["tinyint", "smallint", "int", "bigint", "float", "double", "decimal"]
    float_types = ["float", "double", "decimal"]
    if column_info["dtype"] in numeric_types:
        # Just hardcode the min and max for now
        min = column_info["quartiles"]["min"]
        max = column_info["quartiles"]["max"]
        if column_info["dtype"] in float_types:
            html_template += f""" {min:.2f} → {max:.2f}&nbsp;&nbsp;&nbsp;&nbsp;"""
        else:
            html_template += f""" {int(min)} → {int(max)}&nbsp;&nbsp;&nbsp;&nbsp;"""
        if column_info["num_zeros"] > 0:
            html_template += """ <span class="lightorange"> Zero: <<num_zeros>></span>"""

    # Non-numeric columns get the number of unique values
    else:
        html_template += """ Unique: <<unique>> """

    # Do we have any nulls in this column?
    if column_info["nulls"] > 0:
        html_template += """ <span class="lightred">Null: <<nulls>></span>"""

    # Replace the column name
    html_template = html_template.replace("<<name>>", column_name)

    # Loop through all the details and replace in the template
    for key, value in column_info.items():
        html_template = html_template.replace(f"<<{key}>>", str(value))

    return html_template


def create_markdown(artifact_details: dict) -> str:
    """Create the Markdown for the details/information about the DataSource or the FeatureSet
    Args:
        artifact_details (dict): A dictionary of information about the artifact
    Returns:
        str: A Markdown string
    """

    markdown_template = """
    **Rows:** <<num_rows>>
    <br>**Columns:** <<num_columns>>
    <br>**Created/Mod:** <<created>> / <<modified>>
    <br>**Tags:** <<sageworks_tags>>
    <br>**S3:** <<s3_storage_location>>

    #### Numeric Columns
    <ul class="no-indent-bullets">
    <<numeric_column_details>>
    </ul>

    #### String Columns
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
    if not artifact_details:
        return "No data source details found"

    # Loop through all the details and replace in the template
    for key, value in artifact_details.items():
        # Hack for dates
        if ".000Z" in str(value):
            value = value.replace(".000Z", "").replace("T", " ")
        markdown_template = markdown_template.replace(f"<<{key}>>", str(value))

    # Fill in numeric column details
    column_stats = artifact_details.get("column_stats", {})
    numeric_column_details = ""
    numeric_types = ["tinyint", "smallint", "int", "bigint", "float", "double", "decimal"]
    for column_name, column_info in column_stats.items():
        if column_info["dtype"] in numeric_types:
            column_html = column_info_html(column_name, column_info)
            numeric_column_details += f"<li>{column_html}</li>"

    # Now actually replace the column details in the markdown
    markdown_template = markdown_template.replace("<<numeric_column_details>>", numeric_column_details)

    # For string columns create collapsible sections that show value counts
    string_column_details = ""
    for column_name, column_info in column_stats.items():
        # Skipping any columns that are dtype string
        if column_info["dtype"] != "string" or "value_counts" not in column_info:
            continue

        # Create the column info
        column_html = column_info_html(column_name, column_info)
        column_details = expanding_list.replace("<<column_info>>", column_html)

        # Populate the bullet list (if we have value counts)
        bullet_list = ""
        for value, count in column_info["value_counts"].items():
            bullet_list += f"<li>{value}: {count}</li>"

        # Add the bullet list to the column details
        column_details = column_details.replace("<<bullet_list>>", bullet_list)

        # Add the column details to the markdown
        string_column_details += column_details

    # Now actually replace the column details in the markdown
    markdown_template = markdown_template.replace("<<string_column_details>>", string_column_details)
    return markdown_template


def create(component_id: str, artifact_details: dict) -> dcc.Markdown:
    """Create a Markdown Component details/information about the DataSource or the FeatureSet
    Args:
        component_id (str): The ID of the UI component
        artifact_details (dict): A dictionary of column information
    Returns:
        dcc.Markdown: A Dash Markdown Component
    """

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Markdown(
        id=component_id,
        children=create_markdown(artifact_details),
        dangerously_allow_html=True,
    )
