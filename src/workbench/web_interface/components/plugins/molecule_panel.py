"""Molecule Panel Component (4 Molecule Viewers in a Row)"""

from dash import html

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.web_interface.components.plugins.molecule_viewer import MoleculeViewer


class MoleculePanel(PluginInterface):
    """Molecule Panel Component (4 Molecule Viewers in a Row)"""

    # Initialize this Plugin Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.COMPOUND

    def __init__(self):
        """Initialize the MoleculePanel plugin class"""
        self.component_id = None
        self.molecule_viewers = []
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Molecule Panel Component with 4 Molecule Viewers.

        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Molecule Viewer Components
        """
        self.component_id = component_id
        self.molecule_viewers = [MoleculeViewer() for _ in range(4)]

        # Create a row of 4 MoleculeViewer components
        row_children = [
            viewer.create_component(component_id=f"{self.component_id}-viewer-{i}")
            for i, viewer in enumerate(self.molecule_viewers)
        ]

        # Define the panel container
        self.container = html.Div(
            id=self.component_id,
            children=row_children,
            style={"display": "flex", "flexDirection": "row", "gap": "10px"},
        )

        # Fill in plugin properties
        self.properties = [(f"{viewer.component_id}-header", "children") for viewer in self.molecule_viewers] + [
            (f"{viewer.component_id}-img", "src") for viewer in self.molecule_viewers
        ]

        return self.container

    def update_properties(self, molecules: list) -> list:
        """Update properties for the Molecule Panel.

        Args:
            molecules (list): A list of dictionaries, each containing:
                - compound_id (str): The compound ID
                - smiles (str): The SMILES representation
        Returns:
            list: A list of updated properties for the Molecule Viewers
        """
        if len(molecules) != 4:
            raise ValueError("MoleculePanel requires exactly 4 molecules.")

        updated_properties = []
        for viewer, molecule_data in zip(self.molecule_viewers, molecules):
            updated_properties.extend(
                viewer.update_properties(
                    compound_id=molecule_data["compound_id"],
                    smiles=molecule_data["smiles"],
                )
            )
        return updated_properties


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    molecules = [
        {
            "compound_id": "AQSOL-0001",
            "smiles": "CC(C)C1=CC=C(C=C1)C(=O)O",
        },
        {
            "compound_id": "AQSOL-0002",
            "smiles": "CC(C)C1=CC=C(C=C1)C(=O)O",
        },
        {
            "compound_id": "AQSOL-0003",
            "smiles": "CC(C)C1=CC=C(C=C1)C(=O)O",
        },
        {
            "compound_id": "AQSOL-0004",
            "smiles": "CC(C)C1=CC=C(C=C1)C(=O)O",
        },
    ]
    PluginUnitTest(MoleculePanel, input_data=molecules, theme="dark").run()
