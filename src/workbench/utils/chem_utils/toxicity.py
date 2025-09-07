"""Toxicity detection utilities for molecular compounds"""

from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import FunctionalGroups as FG

# Precompiled SMARTS patterns for custom toxic functional groups
toxic_smarts_patterns = [
    ("C(=S)N"),  # Dithiocarbamate
    ("P(=O)(O)(O)O"),  # Phosphate Ester
    ("[As](=O)(=O)-[OH]"),  # Arsenic Oxide
    ("[C](Cl)(Cl)(Cl)"),  # Trichloromethyl
    ("[Cr](=O)(=O)=O"),  # Chromium(VI)
    ("[N+](C)(C)(C)(C)"),  # Quaternary Ammonium
    ("[Se][Se]"),  # Diselenide
    ("c1c(Cl)c(Cl)c(Cl)c1"),  # Trichlorinated Aromatic Ring
    ("[CX3](=O)[CX4][Cl,Br,F,I]"),  # Halogenated Carbonyl
    ("[P+](C*)(C*)(C*)(C*)"),  # Phosphonium Group
    ("NC(=S)c1c(Cl)cccc1Cl"),  # Chlorobenzene Thiocarbamate
    ("NC(=S)Nc1ccccc1"),  # Phenyl Thiocarbamate
    ("S=C1NCCN1"),  # Thiourea Derivative
]
compiled_toxic_smarts = [Chem.MolFromSmarts(smarts) for smarts in toxic_smarts_patterns]

# Precompiled SMARTS patterns for exemptions
exempt_smarts_patterns = [
    "c1ccc(O)c(O)c1",  # Phenols
]
compiled_exempt_smarts = [Chem.MolFromSmarts(smarts) for smarts in exempt_smarts_patterns]

# Load functional group hierarchy once during initialization
fgroup_hierarchy = FG.BuildFuncGroupHierarchy()


def contains_heavy_metals(mol: Mol) -> bool:
    """
    Check if a molecule contains any heavy metals (broad filter).

    Args:
        mol: RDKit molecule object.

    Returns:
        bool: True if any heavy metals are detected, False otherwise.
    """
    heavy_metals = {"Zn", "Cu", "Fe", "Mn", "Co", "Pb", "Hg", "Cd", "As"}
    return any(atom.GetSymbol() in heavy_metals for atom in mol.GetAtoms())


def halogen_toxicity_score(mol: Mol) -> Tuple[int, int]:
    """
    Calculate the halogen count and toxicity threshold for a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        Tuple[int, int]: (halogen_count, halogen_threshold), where the threshold
        scales with molecule size (minimum of 2 or 20% of atom count).
    """
    # Define halogens and count their occurrences
    halogens = {"Cl", "Br", "I", "F"}
    halogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in halogens)

    # Define threshold: small molecules tolerate fewer halogens
    # Threshold scales with molecule size to account for reasonable substitution
    molecule_size = mol.GetNumAtoms()
    halogen_threshold = max(2, int(molecule_size * 0.2))  # Minimum 2, scaled by 20% of molecule size

    return halogen_count, halogen_threshold


def toxic_elements(mol: Mol) -> Optional[List[str]]:
    """
    Identifies toxic elements or specific forms of elements in a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        Optional[List[str]]: List of toxic elements or specific forms if found, otherwise None.

    Notes:
        Halogen toxicity logic integrates with `halogen_toxicity_score` and scales thresholds
        based on molecule size.
    """
    # Always toxic elements (heavy metals and known toxic single elements)
    always_toxic = {"Pb", "Hg", "Cd", "As", "Be", "Tl", "Sb"}
    toxic_found = set()

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()

        # Check for always toxic elements
        if symbol in always_toxic:
            toxic_found.add(symbol)

        # Conditionally toxic nitrogen (positively charged)
        if symbol == "N" and formal_charge > 0:
            # Exclude benign quaternary ammonium (e.g., choline-like structures)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](C)(C)(C)C")):  # Example benign structure
                continue
            toxic_found.add("N+")

        # Halogen toxicity: Uses halogen_toxicity_score to flag excessive halogenation
        if symbol in {"Cl", "Br", "I", "F"}:
            halogen_count, halogen_threshold = halogen_toxicity_score(mol)
            if halogen_count > halogen_threshold:
                toxic_found.add(symbol)

    return list(toxic_found) if toxic_found else None


def toxic_groups(mol: Chem.Mol) -> Optional[List[str]]:
    """
    Check if a molecule contains known toxic functional groups using RDKit's functional groups and SMARTS patterns.

    Args:
        mol (rdkit.Chem.Mol): The molecule to evaluate.

    Returns:
        Optional[List[str]]: List of SMARTS patterns for toxic groups if found, otherwise None.
    """
    toxic_smarts_matches = []

    # Use RDKit's functional group definitions
    toxic_group_names = ["Nitro", "Azide", "Alcohol", "Aldehyde", "Halogen", "TerminalAlkyne"]
    for group_name in toxic_group_names:
        group_node = next(node for node in fgroup_hierarchy if node.label == group_name)
        if mol.HasSubstructMatch(Chem.MolFromSmarts(group_node.smarts)):
            toxic_smarts_matches.append(group_node.smarts)  # Use group_node's SMARTS directly

    # Check for custom precompiled toxic SMARTS patterns
    for smarts, compiled in zip(toxic_smarts_patterns, compiled_toxic_smarts):
        if mol.HasSubstructMatch(compiled):  # Use precompiled SMARTS
            toxic_smarts_matches.append(smarts)

    # Special handling for N+
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+]")):
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("C[N+](C)(C)C")):  # Exclude benign
            toxic_smarts_matches.append("[N+]")  # Append as SMARTS

    # Exempt stabilizing functional groups using precompiled patterns
    for compiled in compiled_exempt_smarts:
        if mol.HasSubstructMatch(compiled):
            return None

    return toxic_smarts_matches if toxic_smarts_matches else None


if __name__ == "__main__":
    print("Running toxicity detection tests...")

    # Test molecules with descriptions
    test_molecules = {
        # Safe molecules
        "water": ("O", "Water - should be safe"),
        "benzene": ("c1ccccc1", "Benzene - simple aromatic"),
        "glucose": ("C(C1C(C(C(C(O1)O)O)O)O)O", "Glucose - sugar"),
        "ethanol": ("CCO", "Ethanol - simple alcohol"),
        # Heavy metal containing
        "lead_acetate": ("CC(=O)[O-].CC(=O)[O-].[Pb+2]", "Lead acetate - contains Pb"),
        "mercury_chloride": ("Cl[Hg]Cl", "Mercury chloride - contains Hg"),
        "arsenic_trioxide": ("O=[As]O[As]=O", "Arsenic trioxide - contains As"),
        # Halogenated compounds
        "chloroform": ("C(Cl)(Cl)Cl", "Chloroform - trichloromethyl"),
        "ddt": ("c1ccc(cc1)C(c2ccc(cc2)Cl)C(Cl)(Cl)Cl", "DDT - heavily chlorinated"),
        "fluorobenzene": ("Fc1ccccc1", "Fluorobenzene - single halogen"),
        # Nitrogen compounds
        "nitrobenzene": ("c1ccc(cc1)[N+](=O)[O-]", "Nitrobenzene - nitro group"),
        "choline": ("C[N+](C)(C)CCO", "Choline - benign quaternary ammonium"),
        "toxic_quat": ("[N+](C)(C)(C)(C)", "Toxic quaternary ammonium"),
        # Phenol (exempt)
        "catechol": ("c1ccc(O)c(O)c1", "Catechol - phenol, should be exempt"),
        # Phosphate
        "phosphate": ("P(=O)(O)(O)O", "Phosphate ester - toxic pattern"),
    }

    # Test 1: Heavy Metals Detection
    print("\n1. Testing heavy metals detection...")
    for name, (smiles, desc) in test_molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            has_metals = contains_heavy_metals(mol)
            expected = name in ["lead_acetate", "mercury_chloride", "arsenic_trioxide"]
            status = "✓" if has_metals == expected else "✗"
            print(f"   {status} {name}: {has_metals} (expected: {expected})")

    # Test 2: Halogen Toxicity Score
    print("\n2. Testing halogen toxicity scoring...")
    halogen_tests = ["chloroform", "ddt", "fluorobenzene", "benzene"]
    for name in halogen_tests:
        if name in test_molecules:
            smiles, desc = test_molecules[name]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                count, threshold = halogen_toxicity_score(mol)
                print(f"   {name}: {count} halogens, threshold: {threshold}, toxic: {count > threshold}")

    # Test 3: Toxic Elements
    print("\n3. Testing toxic elements detection...")
    for name, (smiles, desc) in test_molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            toxics = toxic_elements(mol)
            if toxics:
                print(f"   ⚠ {name}: {toxics}")
            elif name in ["lead_acetate", "mercury_chloride", "arsenic_trioxide", "chloroform", "ddt"]:
                print(f"   ✗ {name}: Should have detected toxic elements")
            else:
                print(f"   ✓ {name}: No toxic elements (as expected)")

    # Test 4: Toxic Groups
    print("\n4. Testing toxic functional groups...")
    for name, (smiles, desc) in test_molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            groups = toxic_groups(mol)
            if groups:
                print(f"   ⚠ {name}: Found {len(groups)} toxic group(s)")
                for g in groups[:3]:  # Show first 3 patterns
                    print(f"      - {g[:50]}...")
            elif name == "catechol":
                print(f"   ✓ {name}: Exempt (phenol)")
            elif name in ["nitrobenzene", "phosphate", "chloroform", "ethanol"]:
                print(f"   ✗ {name}: Should have detected toxic groups")
            else:
                print(f"   ✓ {name}: No toxic groups")

    # Test 5: Edge Cases
    print("\n5. Testing edge cases...")
    edge_cases = [
        ("", "Empty SMILES"),
        ("INVALID", "Invalid SMILES"),
        ("C" * 100, "Very long carbon chain"),
        ("[N+](C)(C)(C)C", "Benign quaternary ammonium"),
    ]

    for smiles, desc in edge_cases:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            metals = contains_heavy_metals(mol)
            elements = toxic_elements(mol)
            groups = toxic_groups(mol)
            print(f"   {desc}: metals={metals}, elements={elements is not None}, groups={groups is not None}")
        else:
            print(f"   {desc}: Invalid molecule (as expected)")

    print("\n✅ All toxicity detection tests completed!")
