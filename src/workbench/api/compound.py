from dataclasses import dataclass, field
import logging
from typing import List

# Workbench Imports
from workbench.utils.chem_utils import svg_from_smiles


@dataclass
class Compound:
    """Compound: Store details about an individual compound."""

    id: str
    smiles: str = None
    tags: List[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("workbench"), init=False)

    def add_tag(self, tag: str) -> None:
        """Add a single tag to the tags list."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a single tag from the tags list."""
        if tag in self.tags:
            self.tags.remove(tag)

    def add_meta(self, key: str, value) -> None:
        """Add metadata to the Compound."""
        self.meta[key] = value

    def details(self) -> dict:
        """Compound Details

        Returns:
            dict: A dictionary of details about the Compound.
        """
        return {"project": "XYZ", "smiles": self.smiles, "tags": self.tags, "meta": self.meta}

    def image(self, width: int = 300, height: int = 200) -> str:
        """Generate an SVG image of the Compound

        Args:
            width (int, optional): The width of the image (default: 300)
            height (int, optional): The height of the image (default: 200)

        Returns:
            str: The SVG image of the Compound
        """
        from workbench.utils.theme_manager import ThemeManager  # Avoid circular import

        return svg_from_smiles(self.smiles, width, height, background=ThemeManager().background())

    def __str__(self) -> str:
        """User-friendly string representation."""
        str_output = (
            f"Compound({self.id})\n  SMILES: {self.smiles}\n  Tags: {', '.join(self.tags) if self.tags else 'None'}"
        )
        str_output += f"\n  Meta: {self.meta if self.meta else 'None'}"
        return str_output


if __name__ == "__main__":
    # Example usage
    compound = Compound("123")
    compound.smiles = "CCO"
    compound.tags = ["alcohol", "primary"]
    compound.add_meta("toxic_info", {"elements": ["C", "H", "O"], "groups": ["alcohol"]})

    # Print details of the Compound
    print(compound.details())

    # Debugging view (__repr__)
    print(repr(compound))

    # User-friendly view (__str__)
    print(compound)

    # Generate an SVG image of the Compound
    print(compound.image())
