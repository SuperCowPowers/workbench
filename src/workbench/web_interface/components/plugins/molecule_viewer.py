"""A molecule viewer plugin component"""

from dash import html, dcc

# Workbench Imports
from workbench.api.compound import Compound
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.compound_utils import details_to_markdown


class MoleculeViewer(PluginInterface):
    """Molecule Viewer Component"""

    # Initialize this Plugin Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.COMPOUND

    def __init__(self):
        """Initialize the MoleculeViewer plugin class"""
        self.component_id = None

        # Initialize the Theme Manager
        self.theme_manager = ThemeManager()

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Molecule Viewer Component without any data.

        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for Molecule Viewer
        """
        self.component_id = component_id
        self.container = html.Div(
            id=self.component_id,
            children=[
                html.H4(id=f"{self.component_id}-header", children="Compound:"),
                html.Img(
                    id=f"{self.component_id}-img",
                    style={"padding": "0px"},
                ),
                dcc.Markdown(
                    id=f"{self.component_id}-summary", dangerously_allow_html=True, children="**Summary Loading...**"
                ),
            ],
            className="workbench-container",
            style={"padding": "10px"},
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-img", "src"),
            (f"{self.component_id}-summary", "children"),
        ]

        return self.container

    def update_properties(self, compound: Compound, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            compound (Compound): The Compound object
            **kwargs:
                - width (int): The width of the image
                - height (int): The height of the image
        Returns:
            list: A list of the updated property values for the plugin
        """

        # Get the width and height of the image
        width = kwargs.get("width", 450)
        height = kwargs.get("height", 300)

        # Header Text
        header_text = f"Compound: {compound.id}"

        # Create the Molecule Image
        img = compound.image(width, height)

        # Compound Summary
        summary = details_to_markdown(compound)

        # Return the updated property values for this plugin
        return [header_text, img, summary]


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    compound = Compound("AQSOL-0001")
    compound.smiles = "CC(C)C1=CC=C(C=C1)C(=O)O"
    PluginUnitTest(MoleculeViewer, input_data=compound, theme="dark").run()
