"""A molecule viewer plugin component"""

from dash import html

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.chem_utils import img_from_smiles


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
                html.H5(id=f"{self.component_id}-header", children="Compound:"),
                html.Img(id=f"{self.component_id}-img", src=""),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-img", "src"),
        ]
        # self.signals = [(f"{self.component_id}-img", "clickData")]

        return self.container

    def update_properties(self, compound_id: str, smiles: str, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            compound_id (str): The ID of the compound
            smiles (str): SMILES representation of the compound
            **kwargs:
                - tbd
        Returns:
            list: A list of the updated property values for the plugin
        """

        # Header Text
        header_text = f"Compound: {compound_id}"

        # Create the Molecule Image
        img = img_from_smiles(smiles, dark_mode=self.theme_manager.dark_mode())

        # Return the updated property values for this plugin
        return [header_text, img]


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    compound_data = {
        "compound_id": "AQSOL-0001",
        "smiles": "CC(C)C1=CC=C(C=C1)C(=O)O",
    }
    PluginUnitTest(MoleculeViewer, input_data=compound_data, theme="dark").run()
