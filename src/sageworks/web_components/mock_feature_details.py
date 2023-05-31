"""A Component for feature details/information"""
from dash import dcc


def create_markdown(feature_info: dict):
    """Create the Markdown for the Feature Details"""

    # Sort the features
    feature_list = sorted(feature_info.items(), key=lambda x: x[1], reverse=True)

    markdown = ""
    for feature, value in feature_list:
        markdown += (
            f"- **{feature}** ({value}): Cool stuff about descriptor {feature} that we get later "
            "[{feature}](https://www.rdkit.org/docs/source/rdkit.Chem.Crippen.html#rdkit.Chem.Crippen.MolLogP)\n"
        )

    # Add the additional info section
    markdown += """
    ##### Additional Information on Features

    For additional information on the features used in this model please
    see
    - [RDKIT Descriptors](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html)
    - [Mordred Descriptors](https://mordred-descriptor.github.io/documentation/master/descriptors.html)
    """
    return markdown


def create(feature_info: dict) -> dcc.Markdown:
    """Create a Markdown Component for the Feature Details"""

    # Generate a figure and wrap it in a Dash Graph Component
    return dcc.Markdown(id="feature_details", children=create_markdown(feature_info))
