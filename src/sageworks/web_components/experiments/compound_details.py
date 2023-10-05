"""A Component for details/information about DataSources"""
from dash import dcc


def create_markdown(artifact_details: dict) -> str:
    """Create the Markdown for the details/information about the DataSource or the FeatureSet
    Args:
        artifact_details (dict): A dictionary of information about the artifact
    Returns:
        str: A Markdown string
    """

    markdown_template = """
    **Rows:** <<num_rows>>  **Columns:** <<num_columns>>
    <br>**Created/Modified:** <<created>> / <<modified>>
    <br>**Tags:** <<sageworks_tags>>
    <br>**S3:** <<s3_storage_location>>
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
