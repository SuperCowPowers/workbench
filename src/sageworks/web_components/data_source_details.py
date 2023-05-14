"""A Component for details/information about DataSources"""
from dash import dcc


def create_markdown(data_source_details: dict) -> str:
    """Create the Markdown for the details/information about the DataSource
    Args:
        data_source_details (dict): A dictionary of information about the data source
    Returns:
        str: A Markdown string
    """

    # Loop through all the details and create a Markdown string
    markdown = ""
    for key, value in data_source_details.items():
        markdown += f"- **{key}**: {value}\n"
    return markdown


def create(component_id: str, data_source_details: dict) -> dcc.Markdown:
    """Create a Markdown Component details/information about the DataSource
    Args:
        data_source_details (dict): A dictionary of column information
    Returns:
        dcc.Markdown: A Dash Markdown Component
    """

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Markdown(id=component_id, children=create_markdown(data_source_details))
