"""Compound Utility/helper methods"""

import re
from workbench.api.compound import Compound
from workbench.utils.markdown_utils import tag_styling


def details_to_markdown(compound: Compound) -> str:
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
    """Exercise the Compound Utilities"""

    # Create a Compound object
    compound = Compound("123")
    compound.smiles = "CCO"
    compound.tags = ["alcohol", "primary"]
    compound.add_meta("toxic_info", {"elements": ["C", "H", "O"], "groups": ["alcohol"]})

    # Print the compuond markdown
    print(details_to_markdown(compound))
