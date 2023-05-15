"""A Component for details/information about DataSources"""
from dash import dcc


def create_markdown(data_source_details: dict) -> str:
    """Create the Markdown for the details/information about the DataSource
    Args:
        data_source_details (dict): A dictionary of information about the data source
    Returns:
        str: A Markdown string
    """

    markdown_template = """
    - **Rows:** <<num_rows>>
    - **Columns:** <<num_columns>>
    - **Created:** <<created>>
    - **Modified:** <<modified>>
    - **Tags:** <<sageworks_tags>>
    - **S3:** <<s3_storage_location>>

    #### String Column Values
    <<value_count_details>>
    """

    # Loop through all the details and replace in the template
    for key, value in data_source_details.items():
        markdown_template = markdown_template.replace(f"<<{key}>>", str(value))
    markdown_template = markdown_template.replace("<<column_details>>", "tbd")
    return markdown_template


def create(component_id: str, data_source_details: dict) -> dcc.Markdown:
    """Create a Markdown Component details/information about the DataSource
    Args:
        component_id (str): The ID of the UI component
        data_source_details (dict): A dictionary of column information
    Returns:
        dcc.Markdown: A Dash Markdown Component
    """

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Markdown(id=component_id, children=create_markdown(data_source_details))
