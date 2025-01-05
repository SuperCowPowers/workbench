import requests


def get_pubchem_summary_dynamic(cid: int) -> dict:
    """Fetch summary information for a compound from PubChem."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    summary = {"Name": None, "Molecular Formula": None, "Molecular Weight": None, "Toxicity": []}

    # Traverse the JSON dynamically to find relevant fields
    sections = data.get("Record", {}).get("Section", [])
    for section in sections:
        if section.get("TOCHeading") == "Names and Identifiers":
            for sub_section in section.get("Section", []):
                if sub_section.get("TOCHeading") == "Record Title":
                    summary["Name"] = (
                        sub_section.get("Information", [{}])[0]
                        .get("Value", {})
                        .get("StringWithMarkup", [{}])[0]
                        .get("String")
                    )

        if section.get("TOCHeading") == "Chemical and Physical Properties":
            for sub_section in section.get("Section", []):
                if sub_section.get("TOCHeading") == "Molecular Formula":
                    summary["Molecular Formula"] = (
                        sub_section.get("Information", [{}])[0]
                        .get("Value", {})
                        .get("StringWithMarkup", [{}])[0]
                        .get("String")
                    )
                if sub_section.get("TOCHeading") == "Molecular Weight":
                    summary["Molecular Weight"] = (
                        sub_section.get("Information", [{}])[0]
                        .get("Value", {})
                        .get("StringWithMarkup", [{}])[0]
                        .get("String")
                    )

        if section.get("TOCHeading") == "Toxicity":
            for sub_section in section.get("Section", []):
                if "Toxicological Information" in sub_section.get("TOCHeading", ""):
                    for info in sub_section.get("Information", []):
                        toxicity_info = info.get("Value", {}).get("StringWithMarkup", [{}])[0].get("String")
                        summary["Toxicity"].append(toxicity_info)

    # Return the cleaned summary with any found values
    return {k: v for k, v in summary.items() if v}


# Example usage
cid = 240  # Ethanol
summary_info = get_pubchem_summary_dynamic(cid)
print(summary_info)
