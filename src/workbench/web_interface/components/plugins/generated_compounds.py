"""A compound details plugin component"""

from dash import html, dcc
import dash_bootstrap_components as dbc

# Workbench Imports
from workbench.api import Compound
from workbench.utils.chem_utils import svg_from_smiles
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.compound_utils import details_to_markdown
from workbench.utils.ai_summary import AISummary
from workbench.utils.ai_compound_generator import AICompoundGenerator


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

        # Create an instance of the AISummary and AICompoundGenerator classes
        self.ai_summary = AISummary()
        self.ai_compound_generator = AICompoundGenerator()

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
                # Left Column: Header + Molecular Image + Details + Summary
                dbc.Col(
                    [
                        html.H3(id=f"{self.component_id}-header", children="Compound Details Loading..."),
                        html.Img(
                            id=f"{self.component_id}-img",
                            className="workbench-container",
                            style={"padding": "0px", "marginBottom": "30px"},
                        ),
                        dcc.Markdown(
                            id=f"{self.component_id}-details",
                            dangerously_allow_html=True,
                            children="**Details Loading...**",
                        ),
                        dcc.Markdown(
                            id=f"{self.component_id}-summary",
                            dangerously_allow_html=True,
                            children="**Summary Details**",
                        ),
                    ],
                    style={"width": "480px", "flex": "0 0 auto"},
                ),  # Fixed width of 480px
                # Right Column: Generated Compounds
                dbc.Col(
                    [
                        # Row with 5 Img components
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Img(id=f"{self.component_id}-img-{i}"),
                                    className="workbench-container",
                                    style={"margin": "10px"},
                                )
                                for i in range(5)
                            ]
                        ),
                        dcc.Markdown(
                            id=f"{self.component_id}-generative",
                            dangerously_allow_html=True,
                            children="**Generative Details**",
                        ),
                    ],
                    style={"flex": "1"},
                ),  # Takes up remaining space
            ],
            style={"display": "flex"},  # Ensures the columns are side by side
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-img", "src"),
            (f"{self.component_id}-details", "children"),
            (f"{self.component_id}-summary", "children"),
            (f"{self.component_id}-generative", "children"),
            (f"{self.component_id}-img-0", "src"),
            (f"{self.component_id}-img-1", "src"),
            (f"{self.component_id}-img-2", "src"),
            (f"{self.component_id}-img-3", "src"),
            (f"{self.component_id}-img-4", "src"),
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

        # Compound Details
        details = details_to_markdown(compound)

        # AI Summary for this compound
        ai_summary_markdown = self.ai_summary.smiles_query(compound.smiles)
        ai_summary_markdown = "#### Summary\n" + ai_summary_markdown

        # AI Compound Generation for this compound
        ai_compound_markdown = self.ai_compound_generator.generate_variants(compound.smiles)
        ai_compound_markdown = "\n#### Generated Compounds\n" + ai_compound_markdown

        # Generate 5 compounds images
        images = []
        generated_smiles = self.ai_compound_generator.get_smiles()
        for i, smiles in enumerate(generated_smiles[:5]):
            img_src = svg_from_smiles(smiles, 300, 200, background=self.theme_manager.background())
            images.append(img_src)

        # Return the updated property values for this plugin
        return [header_text, img, details, ai_summary_markdown, ai_compound_markdown, *images]


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
