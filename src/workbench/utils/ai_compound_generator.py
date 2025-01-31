import os
import re
import logging
from openai import OpenAI
from rdkit import Chem

# Workbench Imports
from workbench.utils.workbench_cache import WorkbenchCache


class AICompoundGenerator:
    def __init__(self):
        """
        Initialize the AICompoundGenerator class by setting up the OpenAI client.
        """
        self.log = logging.getLogger("workbench")

        # Get the API key from the environment
        self.deepseek_api = os.getenv("DEEPSEEK_API")

        # If the API key is not set, raise an error
        if not self.deepseek_api:
            raise ValueError("DEEPSEEK API key is not set")

        # Create a client with the API key
        self.client = OpenAI(api_key=self.deepseek_api, base_url="https://api.deepseek.com/v1")
        self.ai_cache = WorkbenchCache(prefix="ai_compound_generator")
        self.generated_markdown = ""

    def generate_variants(self, smiles_string: str, force_pull=False) -> str:
        """
        Generate new compound variants optimized for specific properties.

        Args:
            smiles_string (str): The SMILES string of the compound.
            force_pull (bool, optional): Force the API to pull the data again. Defaults to False.

        Returns:
            str: A markdown-formatted list of optimized compound variants and their properties.
        """
        # Check if the variants are already cached
        cached_markdown = self.ai_cache.get(f"{smiles_string}_variants")
        if cached_markdown and not force_pull:
            self.log.info(f"Using cached variant markdown for SMILES: {smiles_string}")
            self.generated_markdown = cached_markdown
            return cached_markdown

        # If the variants are not cached, query the API
        self.log.info(f"Generating variants for SMILES: {smiles_string}")

        # Define the task and lookup context
        task = """
            Generate 5 novel compound variants based on the input SMILES string.
            Optimize the variants for the following properties:
            - Reduced Toxicity
            - Increased Solubility
            - Improved Binding Affinity
            - Enhanced Metabolic Stability
            For each variant, provide:
            - The SMILES string
            - A brief explanation of the modifications made
            - The predicted impact on the target properties
            DO NOT INCLUDE INTRODUCTORY SENTENCES, CONCLUSIONS, OR CONVERSATIONAL PHRASES
        """
        lookup_context = {
            "action": "generate_variants",
            "input": smiles_string,
            "optimization_criteria": [
                "reduce toxicity",
                "increase solubility",
                "improve binding affinity",
                "enhance metabolic stability",
            ],
            "query": task,
        }

        try:
            # Make the API request
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # Specify the model
                messages=[{"role": "user", "content": str(lookup_context)}],  # Convert dict to string
            )

            # Extract the variants from the response
            self.generated_markdown = response.choices[0].message.content

            # Cache the generated marked for future use
            self.ai_cache.set(f"{smiles_string}_variants", self.generated_markdown)
            return self.generated_markdown

        except Exception as e:
            # Handle any errors that occur during the API request
            return f"### Error\n\nAn error occurred while generating variants: {str(e)}"

    @staticmethod
    def escape_markdown(value) -> str:
        """Escape special characters in Markdown strings."""
        return re.sub(r"([<>\[\]])", r"\\\1", str(value))

    def extract_smiles_and_desc(self) -> list[tuple[str, str]]:
        """
        Extracts SMILES strings and their corresponding descriptions from our generated markdown.

        Returns:
            list[tuple[str, str]]: A list of tuples, where each tuple contains a valid SMILES string
                                   and its corresponding description (with the SMILES as the first line).
        """
        # Split the markdown into individual entries based on the **SMILES**: keyword
        entries = re.split(r"\n\s*\d+\.\s\*\*SMILES\*\*:", self.generated_markdown)

        # Remove the first entry if it's empty (due to the split)
        if entries and not entries[0].strip():
            entries = entries[1:]

        # Process each entry
        results = []
        for entry in entries:
            # Extract SMILES, Explanation, and Impact using a more flexible regex
            match = re.match(
                r"\s*([^\n]+)\n\s*\*\*Explanation\*\*:\s*([^\n]+)\n\s*\*\*Impact\*\*:\s*([\s\S]+)",
                entry,
            )
            if match:
                smiles, explanation, impact = match.groups()
                smiles = smiles.strip()
                if self.is_valid_smiles(smiles):
                    explanation = explanation.strip()
                    impact = "\n".join([line.strip() for line in impact.split("\n")])
                    description = (
                        f"**SMILES**: {self.escape_markdown(smiles)}<br>"
                        f"**Explanation**: {explanation}<br>"
                        f"**Impact**:<br>" + impact.replace("- ", "-&nbsp;").replace("\n", "<br>").replace("**", "")
                    )
                    results.append((smiles, description))

        return results

    @staticmethod
    def is_valid_smiles(smiles: str) -> bool:
        """
        Checks if a SMILES string is valid using RDKit.

        Args:
            smiles (str): The SMILES string to validate.

        Returns:
            bool: True if the SMILES string is valid, False otherwise.
        """
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None


if __name__ == "__main__":
    # Example SMILES from tox21 dataset
    smiles_string = "CCCCCCCCCCCCCC[n+]1ccc(C)cc1.[Cl-]"
    # smiles_string = "CCCC[n+]1cccc(C)c1.F[B-](F)(F)F"
    smiles_string = "CCCCCCCCCCCCCC[n+]1ccc(C)cc1.[Cl-]"

    # Create an instance of the AICompoundGenerator class
    compound_generator = AICompoundGenerator()

    # Generate variants and get the markdown result
    markdown_result = compound_generator.generate_variants(smiles_string, force_pull=False)

    # Print the markdown result
    print(markdown_result)

    # Get the SMILES strings and descriptions from the markdown
    smile_descriptions = compound_generator.extract_smiles_and_desc()
    for smiles, description in smile_descriptions:
        print(f"SMILES: {smiles}")
        print(f"Description: {description}")
        print()
