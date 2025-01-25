import os
import logging
from openai import OpenAI
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
        cached_variants = self.ai_cache.get(f"{smiles_string}_variants")
        if cached_variants and not force_pull:
            self.log.info(f"Using cached variants for SMILES: {smiles_string}")
            return cached_variants

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
            variants = response.choices[0].message.content

            # Cache the variants for future use
            self.ai_cache.set(f"{smiles_string}_variants", variants)
            return variants

        except Exception as e:
            # Handle any errors that occur during the API request
            return f"### Error\n\nAn error occurred while generating variants: {str(e)}"


if __name__ == "__main__":
    # Example SMILES from tox21 dataset
    smiles_string = "CCCCCCCCCCCCCC[n+]1ccc(C)cc1.[Cl-]"

    # Create an instance of the AICompoundGenerator class
    compound_generator = AICompoundGenerator()

    # Generate variants and get the markdown result
    markdown_result = compound_generator.generate_variants(smiles_string, force_pull=True)

    # Print the markdown result
    print(markdown_result)
