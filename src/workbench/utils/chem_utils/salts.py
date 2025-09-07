"""Salt feature extraction utilities for molecular analysis"""

import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union

# Molecular Descriptor Imports
from rdkit import Chem
from rdkit.Chem import Descriptors

# Set up the logger
log = logging.getLogger("workbench")


def _get_salt_feature_columns() -> List[str]:
    """Internal: Return list of all salt feature column names"""
    return [
        "has_salt",
        "mw_ratio",
        "salt_to_api_ratio",
        "has_metal_salt",
        "has_halide",
        "ionic_strength_proxy",
        "has_organic_salt",
    ]


def _classify_salt_types(salt_frags: List[Chem.Mol]) -> Dict[str, int]:
    """Internal: Classify salt fragments into categories"""
    features = {
        "has_organic_salt": 0,
        "has_metal_salt": 0,
        "has_halide": 0,
    }

    for frag in salt_frags:
        # Get atoms
        atoms = [atom.GetSymbol() for atom in frag.GetAtoms()]

        # Metal detection
        metals = ["Na", "K", "Ca", "Mg", "Li", "Zn", "Fe", "Al"]
        if any(metal in atoms for metal in metals):
            features["has_metal_salt"] = 1

        # Halide detection
        halides = ["Cl", "Br", "I", "F"]
        if any(halide in atoms for halide in halides):
            features["has_halide"] = 1

        # Organic vs inorganic (simple heuristic: contains C)
        if "C" in atoms:
            features["has_organic_salt"] = 1

    return features


def extract_advanced_salt_features(
    mol: Optional[Chem.Mol],
) -> Tuple[Optional[Dict[str, Union[int, float]]], Optional[Chem.Mol]]:
    """Extract comprehensive salt-related features from RDKit molecule"""
    if mol is None:
        return None, None

    # Get fragments
    fragments = Chem.GetMolFrags(mol, asMols=True)

    # Identify API (largest organic fragment) vs salt fragments
    fragment_weights = [(frag, Descriptors.MolWt(frag)) for frag in fragments]
    fragment_weights.sort(key=lambda x: x[1], reverse=True)

    # Find largest organic fragment as API
    api_mol = None
    salt_frags = []

    for frag, mw in fragment_weights:
        atoms = [atom.GetSymbol() for atom in frag.GetAtoms()]
        if "C" in atoms and api_mol is None:  # First organic fragment = API
            api_mol = frag
        else:
            salt_frags.append(frag)

    # Fallback: if no organic fragments, use largest
    if api_mol is None:
        api_mol = fragment_weights[0][0]
        salt_frags = [frag for frag, _ in fragment_weights[1:]]

    # Initialize all features with default values
    features = {col: 0 for col in _get_salt_feature_columns()}
    features["mw_ratio"] = 1.0  # default for no salt

    # Basic features
    features.update(
        {
            "has_salt": int(len(salt_frags) > 0),
            "mw_ratio": Descriptors.MolWt(api_mol) / Descriptors.MolWt(mol),
        }
    )

    if salt_frags:
        # Salt characterization
        total_salt_mw = sum(Descriptors.MolWt(frag) for frag in salt_frags)
        features.update(
            {
                "salt_to_api_ratio": total_salt_mw / Descriptors.MolWt(api_mol),
                "ionic_strength_proxy": sum(abs(Chem.GetFormalCharge(frag)) for frag in salt_frags),
            }
        )

        # Salt type classification
        features.update(_classify_salt_types(salt_frags))

    return features, api_mol


def add_salt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add salt features to dataframe with 'molecule' column containing RDKit molecules"""
    salt_features_list = []

    for idx, row in df.iterrows():
        mol = row["molecule"]
        features, clean_mol = extract_advanced_salt_features(mol)

        if features is None:
            # Handle invalid molecules
            features = {col: None for col in _get_salt_feature_columns()}

        salt_features_list.append(features)

    # Convert to DataFrame and concatenate
    salt_df = pd.DataFrame(salt_features_list)
    return pd.concat([df, salt_df], axis=1)


if __name__ == "__main__":
    print("Running salt feature extraction tests...")

    # Test molecules with various salt forms
    test_molecules = {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",  # No salt
        "sodium_acetate": "CC(=O)[O-].[Na+]",  # Simple sodium salt
        "calcium_acetate": "CC(=O)[O-].[Ca+2].CC(=O)[O-]",  # Calcium salt (divalent)
        "ammonium_chloride": "[NH4+].[Cl-]",  # Inorganic salt
        "methylamine_hcl": "CN.Cl",  # Organic salt
        "benzene": "c1ccccc1",  # No salt
        "aspirin_sodium": "CC(=O)OC1=CC=CC=C1C(=O)[O-].[Na+]",  # API with sodium salt
        "complex_salt": "CC(C)(C)c1ccc(O)cc1.Cl.Cl.[Zn+2]",  # Complex with metal
    }

    # Test 1: Basic salt feature extraction
    print("\n1. Testing basic salt feature extraction...")

    salt_test_df = pd.DataFrame({"smiles": list(test_molecules.values()), "name": list(test_molecules.keys())})

    salt_test_df["molecule"] = salt_test_df["smiles"].apply(Chem.MolFromSmiles)
    salt_result = add_salt_features(salt_test_df)

    print("   Salt feature results:")
    print(f"   {'Molecule':20} has_salt  metal  halide  organic  mw_ratio")
    print("   " + "-" * 65)
    for _, row in salt_result.iterrows():
        print(
            f"   {row['name']:20} {row['has_salt']:^8} {row['has_metal_salt']:^6} "
            f"{row['has_halide']:^7} {row['has_organic_salt']:^8} {row['mw_ratio']:>8.3f}"
        )

    # Test 2: Detailed feature extraction for specific cases
    print("\n2. Testing detailed salt feature extraction...")

    for name, smiles in [
        ("sodium_acetate", test_molecules["sodium_acetate"]),
        ("calcium_acetate", test_molecules["calcium_acetate"]),
        ("aspirin", test_molecules["aspirin"]),
    ]:
        mol = Chem.MolFromSmiles(smiles)
        features, api_mol = extract_advanced_salt_features(mol)

        print(f"\n   {name}:")
        print(f"     Total MW: {Descriptors.MolWt(mol):.2f}")
        if api_mol:
            print(f"     API MW: {Descriptors.MolWt(api_mol):.2f}")
        if features:
            print(f"     Salt/API ratio: {features['salt_to_api_ratio']:.3f}")
            print(f"     Ionic strength proxy: {features['ionic_strength_proxy']}")

    # Test 3: Edge cases
    print("\n3. Testing edge cases...")

    # Empty molecule (None)
    empty_features, empty_api = extract_advanced_salt_features(None)
    print(f"   None molecule: features={empty_features}, api={empty_api}")

    # Single fragment (no salt)
    benzene_mol = Chem.MolFromSmiles("c1ccccc1")
    benzene_features, benzene_api = extract_advanced_salt_features(benzene_mol)
    print(
        f"   Single fragment (benzene): has_salt={benzene_features['has_salt']}, "
        f"mw_ratio={benzene_features['mw_ratio']:.3f}"
    )

    # Multiple organic fragments
    multi_org = Chem.MolFromSmiles("c1ccccc1.CC(=O)O")
    multi_features, multi_api = extract_advanced_salt_features(multi_org)
    print(f"   Multiple organic fragments: has_salt={multi_features['has_salt']}")

    # Test 4: Salt type classification
    print("\n4. Testing salt type classification...")

    test_salts = [
        ("NaCl", "[Na+].[Cl-]", "Sodium chloride"),
        ("KBr", "[K+].[Br-]", "Potassium bromide"),
        ("CaCl2", "[Ca+2].[Cl-].[Cl-]", "Calcium chloride"),
        ("Organic salt", "CC(=O)[O-].CN", "Acetate with methylamine"),
    ]

    for name, smiles, description in test_salts:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            features, _ = extract_advanced_salt_features(mol)
            print(f"   {name:15} ({description})")
            print(
                f"     Metal: {features['has_metal_salt']}, "
                f"Halide: {features['has_halide']}, "
                f"Organic: {features['has_organic_salt']}"
            )

    # Test 5: DataFrame integration
    print("\n5. Testing DataFrame integration...")

    # Create a mixed DataFrame
    mixed_df = pd.DataFrame(
        {
            "smiles": [
                test_molecules["aspirin"],
                test_molecules["sodium_acetate"],
                test_molecules["calcium_acetate"],
            ],
            "name": ["aspirin", "sodium_acetate", "calcium_acetate"],
            "existing_col": [1, 2, 3],  # Test that existing columns are preserved
        }
    )

    mixed_df["molecule"] = mixed_df["smiles"].apply(Chem.MolFromSmiles)
    result_df = add_salt_features(mixed_df)

    # Check that all columns are present
    expected_cols = ["smiles", "name", "existing_col", "molecule"] + _get_salt_feature_columns()
    missing_cols = [col for col in expected_cols if col not in result_df.columns]

    if missing_cols:
        print(f"   ✗ Missing columns: {missing_cols}")
    else:
        print("   ✓ All expected columns present")
        print(f"   ✓ Original columns preserved: 'existing_col' in result = {('existing_col' in result_df.columns)}")
        print(f"   ✓ Salt features added: {len(_get_salt_feature_columns())} new columns")

    print("\n✅ All salt feature tests completed!")
