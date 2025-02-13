"""A generated compounds plugin component"""

from dash import html, dcc
import dash_bootstrap_components as dbc

# Workbench Imports
from workbench.api.compound import Compound
from workbench.utils.chem_utils import svg_from_smiles
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.compound_utils import details_to_markdown
from workbench.utils.ai_summary import AISummary
from workbench.utils.ai_compound_generator import AICompoundGenerator


class GeneratedCompounds(PluginInterface):
    """Generated Compound Component"""

    # Initialize this Plugin Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.COMPOUND

    def __init__(self):
        """Initialize the GeneratedCompounds plugin class"""
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
                # Left Column: Molecule Viewer + Summary
                dbc.Col(
                    [
                        html.H4(id=f"{self.component_id}-header", children="Compound:"),
                        html.Img(
                            id=f"{self.component_id}-img",
                            style={"padding": "0px"},
                            className="workbench-highlight",
                        ),
                        dcc.Markdown(
                            id=f"{self.component_id}-summary",
                            dangerously_allow_html=True,
                            children="**Summary Details**",
                            style={"margin-top": "20px"},
                        ),
                    ],
                    style={"width": "480px", "flex": "0 0 auto", "marginRight": 30},  # Fixed width of 480px
                ),
                # Right Column: 5 Rows of Generated Compounds
                dbc.Col(
                    [
                        # Add H3 header at the top
                        dbc.Row(
                            html.H3("Generated Compounds", style={"marginBottom": "10px"})
                        ),
                        # Create 5 rows, each with an image and a Markdown component
                        *[
                            dbc.Row(
                                [
                                    # Fixed-width image column (300px)
                                    dbc.Col(
                                        html.Img(
                                            id=f"{self.component_id}-img-{i}",
                                            style={"margin": "10px", "maxWidth": "100%", "height": "auto", "width": "300px"},
                                        ),
                                        style={"flex": "0 0 300px"},  # Fixed width of 300px
                                    ),
                                    # Flex markdown column (takes remaining space)
                                    dbc.Col(
                                        dcc.Markdown(
                                            id=f"{self.component_id}-markdown-{i}",
                                            dangerously_allow_html=True,
                                            children=f"**Details for Image {i + 1}**",
                                        ),
                                        style={"flex": "1", "paddingTop": "10px"},
                                    ),
                                ],
                                style={"marginBottom": "10px", "display": "flex"}, className="workbench-offset",
                            )
                            for i in range(5)
                        ],
                    ],
                    style={"flex": "1"},
                )
            ],
            style={"display": "flex"},  # Flexbox layout
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-img", "src"),
            (f"{self.component_id}-summary", "children"),
            (f"{self.component_id}-img-0", "src"),
            (f"{self.component_id}-markdown-0", "children"),
            (f"{self.component_id}-img-1", "src"),
            (f"{self.component_id}-markdown-1", "children"),
            (f"{self.component_id}-img-2", "src"),
            (f"{self.component_id}-markdown-2", "children"),
            (f"{self.component_id}-img-3", "src"),
            (f"{self.component_id}-markdown-3", "children"),
            (f"{self.component_id}-img-4", "src"),
            (f"{self.component_id}-markdown-4", "children"),
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
        width = kwargs.get("width", 300)
        height = kwargs.get("height", 200)

        # Header Text
        header_text = f"Compound: {compound.id}"

        # Generate a molecule image for the compound
        img = compound.image(450, 300)

        # AI Summary for this compound
        ai_summary_markdown = self.ai_summary.smiles_query(compound.smiles)
        ai_summary_markdown = "#### Summary<br>" + ai_summary_markdown

        # AI Compound Generation for this compound
        _ai_compound_markdown = self.ai_compound_generator.generate_variants(compound.smiles)

        # Generate 5 rows, each with an image and markdown details
        generated_smiles_and_desc = self.ai_compound_generator.extract_smiles_and_desc()

        # Sanity check that we have at least 5 generated compounds
        if len(generated_smiles_and_desc) < 5:
            generated_smiles_and_desc += [
                (compound.smiles, f"Compound Generation failed for {compound.smiles}") for _ in range(5 - len(generated_smiles_and_desc))
            ]
        image_markdown_pairs = [
            (
                svg_from_smiles(smiles, 300, 200, background=self.theme_manager.background()),
                description,
            )
            for smiles, description in generated_smiles_and_desc
        ]
        # Flatten the list of tuples into a single list
        flattened_pairs = [item for pair in image_markdown_pairs for item in pair]

        # Return the updated property values for this plugin
        return [header_text, img, ai_summary_markdown, *flattened_pairs]


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    compound = Compound("XYZ-0001")
    compound.smiles = "CC(C)C1=CC=C(C=C1)C(=O)O"
    compound.smiles = "CCCC[n+]1cccc(C)c1.F[B-](F)(F)F"
    compound.smiles = "CCCCCCCCCCCCCC[n+]1ccc(C)cc1.[Cl-]"
    compound.tags = ["toxic", "primary"]
    compound.meta = {"toxic_elements": None, "toxic_groups": ["[C;$(C#CH)]", "[C;$(C#CH)]"]}
    PluginUnitTest(GeneratedCompounds, input_data=compound, theme="light").run()
