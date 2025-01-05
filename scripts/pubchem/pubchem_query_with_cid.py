import pubchempy as pcp
import requests

# Define the SMILES string
smiles = "CCO"  # Example: Ethanol

# Search for compounds matching the SMILES string
compounds = pcp.get_compounds(smiles, namespace="smiles")

# Use the CID obtained from the previous step
cid = compounds[0].cid

# Define the URL for the Safety and Hazards section
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Safety%20and%20Hazards"

# Make the API request
response = requests.get(url)
data = response.json()

# Extract and print the Safety and Hazards information
# (Parsing the JSON structure is required to extract specific details)
print(data)
