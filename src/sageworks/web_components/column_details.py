"""A Component for details/information about each column in the DataSource or FeatureSet"""
from dash import dcc


def create_markdown(column_info: dict) -> str:
    """Create the Markdown for the details/information about each column
    Args:
        column_info (dict): A dictionary of column information
    Returns:
        str: A markdown string
    """

    # Loop through each column and create a markdown entry
    markdown = ""
    for column, info in column_info.items():
        markdown += (
            f"- **{column}** ({info}):"
        )
    return markdown


def create(column_info: dict) -> dcc.Markdown:
    """Create a Markdown Component details/information about the DataSource or FeatureSet"""

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Markdown(id="column_details", children=create_markdown(column_info))
