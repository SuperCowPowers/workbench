import pubchempy as pcp

# Define the SMILES string
smiles = "CCO"  # Example: Ethanol

# Search for compounds matching the SMILES string
compounds = pcp.get_compounds(smiles, namespace="smiles")

# Check if any compounds were found
if compounds:
    compound = compounds[0]  # Take the first matching compound

    # Retrieve desired properties
    compound_info = {
        "Name": compound.iupac_name,
        "Molecular Formula": compound.molecular_formula,
        "Molecular Weight": compound.molecular_weight,
        "CID": compound.cid,
    }

    # Print the compound information
    print(compound_info)
else:
    print("No compound found for the given SMILES string.")
