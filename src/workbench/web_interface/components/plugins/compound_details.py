"""A compound details plugin component"""

import re
from dash import html, dcc

# Workbench Imports
from workbench.api import Compound
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.chem_utils import svg_from_smiles
from workbench.utils.markdown_utils import tag_styling


class CompoundDetails(PluginInterface):
    """Molecule Viewer Component"""

    # Initialize this Plugin Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.COMPOUND

    def __init__(self):
        """Initialize the CompoundDetails plugin class"""
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
                html.H5(id=f"{self.component_id}-header", children="Compound Details Loading..."),
                html.Img(
                    id=f"{self.component_id}-img",
                    className="workbench-container",
                    style={"padding": "0px", "marginBottom": "30px"},
                ),
                html.H5("Details:"),
                dcc.Markdown(id=f"{self.component_id}-details", dangerously_allow_html=True),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-img", "src"),
            (f"{self.component_id}-details", "children"),
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
        img = svg_from_smiles(compound.smiles, width, height, background=self.theme_manager.background())

        # Compound Details
        details = self.details_to_markdown(compound)

        # Return the updated property values for this plugin
        return [header_text, img, details]

    def details_to_markdown(self, compound: Compound) -> str:
        """Construct the markdown string for compound details.

        Args:
            compound (Compound): The Compound object

        Returns:
                str: A markdown string
        """

        def escape_markdown(value) -> str:
            """Escape special characters in Markdown strings."""
            return re.sub(r"([<>\[\]])", r"\\\1", str(value))

        # Construct the markdown string
        markdown = ""
        for key, value in compound.details().items():
            # Convert tags to styled spans
            if key == "tags":
                tag_substrings = {"toxic": "alert", "heavy": "warning", "frag": "warning", "druglike": "good"}
                markdown += f"**{key}:** {tag_styling(value, tag_substrings)}  \n"
            # For dictionaries, convert to Markdown
            elif isinstance(value, dict):
                for k, v in value.items():
                    if v is not None:
                        if isinstance(v, list):
                            v = ", ".join(v)
                        escaped_value = escape_markdown(v)
                        markdown += f"**{k}:** {escaped_value}  \n"
            else:
                escaped_value = escape_markdown(value)
                markdown += f"**{key}:** {escaped_value}  \n"

        return markdown


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    # Run the Unit Test on the Plugin
    compound = Compound("AQSOL-0001")
    compound.smiles = "CC(C)C1=CC=C(C=C1)C(=O)O"
    compound.tags = ["toxic", "primary"]
    compound.meta = {"toxic_elements": None, "toxic_groups": ["[C;$(C#CH)]", "[C;$(C#CH)]"]}
    PluginUnitTest(CompoundDetails, input_data=compound, theme="light").run()
