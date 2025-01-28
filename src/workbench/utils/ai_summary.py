import os
import re
import logging
from openai import OpenAI

from workbench.utils.workbench_cache import WorkbenchCache


class AISummary:
    def __init__(self):
        """
        Initialize the AISummary class by setting up the OpenAI client.
        """
        self.log = logging.getLogger("workbench")

        # Get the API key from the environment
        self.deepseek_api = os.getenv("DEEPSEEK_API")

        # If the API key is not set, raise an error
        if not self.deepseek_api:
            raise ValueError("DEEPSEEK API key is not set")

        # Create a client with the API key
        self.client = OpenAI(api_key=self.deepseek_api, base_url="https://api.deepseek.com/v1")
        self.ai_cache = WorkbenchCache(prefix="ai_summary")

    def smiles_query(self, smiles_string: str, force_pull=False) -> str:
        """
        Query the DeepSeek API for information about a compound using its SMILES string.

        Args:
            smiles_string (str): The SMILES string of the compound.
            force_pull (bool, optional): Force the API to pull the data again. Defaults to False.

        Returns:
            str: A markdown-formatted bulleted list of the compound's properties.
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
            Search open sources and use internal chemistry knowledge to provide a concise
            summary of its properties relevant to targeted therapeutics and toxicity.
            DO NOT INCLUDE INTRODUCTORY SENTENCES, CONCLUSIONS, OR CONVERSATIONAL PHRASES
        """
        lookup_context = {
            "action": "lookup",
            "resources": ["PubMed", "PubChem", "ChEMBL", "DrugBank", "ChemSpider", "open source"],
            "query": f"Find information about the compound with SMILES {smiles_string}. {task}",
        }

        try:
            # Make the API request
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # Specify the model
                messages=[{"role": "user", "content": str(lookup_context)}],  # Convert dict to string
            )

            # Extract the summary from the response
            summary = response.choices[0].message.content

            # If one of the lines has SMILES, delete that line
            summary = "\n".join([line for line in summary.split("\n") if "SMILES" not in line])

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
    # smiles_string = "CCCCCCCCCCCCCC[n+]1ccc(C)cc1.[Cl-]"

    # Create an instance of the AISummary class
    ai_summary = AISummary()

    # Query the API and get the markdown summary
    summary_markdown = ai_summary.smiles_query(smiles_string, force_pull=True)

    # Print the markdown result
    print(summary_markdown)
