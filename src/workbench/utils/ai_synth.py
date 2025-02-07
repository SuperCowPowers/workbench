import os
import re
import logging
from openai import OpenAI

from workbench.utils.workbench_cache import WorkbenchCache

use_openai = False


class AISynth:
    def __init__(self):
        """
        Initialize the AISynth class by setting up the OpenAI client.
        """
        self.log = logging.getLogger("workbench")

        # Get the API key from the environment
        if use_openai:
            self.api_key = os.getenv("OPENAI_API")
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-4o"
        else:
            self.api_key = os.getenv("DEEPSEEK_API")
            self.base_url = "https://api.deepseek.com/v1"
            self.model = "deepseek-chat"

        # If the API key is not set, raise an error
        if not self.api_key:
            raise ValueError("AI API key is not set")

        # Create a client with the API key
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.ai_cache = WorkbenchCache(prefix="ai_synth")

    def smiles_query(self, smiles_string: str, force_pull=False) -> str:
        """
        Query the DeepSeek API for information about a compound using its SMILES string.

        Args:
            smiles_string (str): The SMILES string of the compound.
            force_pull (bool, optional): Force the API to pull the data again. Defaults to False.

        Returns:
            str: A markdown-formatted response containing a synthetic accessibility score (1-10)
                 and a description of how to synthesize the compound.
        """
        # Check if the summary is already cached
        cached_summary = self.ai_cache.get(smiles_string)
        if cached_summary and not force_pull:
            self.log.info(f"Using cached summary for SMILES: {smiles_string}")
            return cached_summary

        # If the summary is not cached, query the API
        self.log.info(f"Querying the API for SMILES: {smiles_string}")

        # Define the task and lookup context
        task = """
            Provide the following information for the compound with the given SMILES string:
            1. A synthetic accessibility score (1-10), where 1 is very easy to synthesize and 10 is very difficult.
            2. A concise description of how to synthesize the compound, including common precursors and reaction steps.
            DO NOT INCLUDE INTRODUCTORY SENTENCES, CONCLUSIONS, OR CONVERSATIONAL PHRASES.
        """
        lookup_context = {
            "action": "lookup",
            "resources": ["PubMed", "PubChem", "ChEMBL", "DrugBank", "ChemSpider", "open source"],
            "query": f"Provide synthetic accessibility information for compound {smiles_string}. {task}",
        }

        try:
            # Make the API request
            response = self.client.chat.completions.create(
                model=self.model,  # Specify the model
                messages=[{"role": "user", "content": str(lookup_context)}],  # Convert dict to string
            )

            # Extract the summary from the response
            summary = response.choices[0].message.content

            # Clean up the summary
            # summary = self.clean_summary(summary)

            # Cache the summary for future use
            self.ai_cache.set(smiles_string, summary)
            return summary

        except Exception as e:
            # Handle any errors that occur during the API request
            return f"### Error\n\nAn error occurred while querying the API: {str(e)}"

    @staticmethod
    def clean_summary(response: str) -> str:
        # Remove leading numbers with dots (e.g., "1. ", "2. ")
        response = re.sub(r"^\d+\.\s", "", response, flags=re.MULTILINE)

        # Strip whitespace from each line
        cleaned_lines = [line.strip() for line in response.split("\n")]

        # Join the cleaned lines into a single string
        cleaned_response = "\n".join(cleaned_lines)

        return cleaned_response


# Example usage
if __name__ == "__main__":
    # Example SMILES from tox21 dataset
    smiles_string = "CC(C)C1=CC=C(C=C1)C(=O)O"
    smiles_string = "CCCC[n+]1cccc(C)c1.F[B-](F)(F)F"
    smiles_string = "CCCC[n+]1ccc(C)cc1.[I-]"
    # smiles_string = "CCCCCCCCCCCCCC[n+]1ccc(C)cc1.[Cl-]"

    # Create an instance of the AISynth class
    ai_summary = AISynth()

    # Query the API and get the markdown summary
    summary_markdown = ai_summary.smiles_query(smiles_string, force_pull=True)

    # Print the markdown result
    print(summary_markdown)
