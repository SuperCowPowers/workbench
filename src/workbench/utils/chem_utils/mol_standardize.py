"""
mol_standardize.py - Molecular Standardization for ADMET Preprocessing
Following ChEMBL structure standardization pipeline

Purpose:
    Standardizes chemical structures to ensure consistent molecular representations
    for ADMET modeling. Handles tautomers, salts, charges, and structural variations
    that can cause the same compound to be represented differently.

Standardization Pipeline:
    1. Cleanup
       - Removes explicit hydrogens
       - Disconnects metal atoms from organic fragments
       - Normalizes functional groups (e.g., nitro, sulfoxide representations)

    2. Fragment Parent Selection (optional, controlled by extract_salts parameter)
       - Identifies and keeps the largest organic fragment
       - Removes salts, solvents, and counterions
       - Example: [Na+].CC(=O)[O-] → CC(=O)O (keeps acetate, removes sodium)

    3. Charge Neutralization (optional, controlled by extract_salts parameter)
       - Neutralizes charges where possible
       - Only applied when extract_salts=True (following ChEMBL pipeline)
       - Skipped when extract_salts=False to preserve ionic character
       - Example: CC(=O)[O-] → CC(=O)O

    4. Tautomer Canonicalization (optional, default=True)
       - Generates canonical tautomer form for consistency
       - Example: Oc1ccccn1 → O=c1cccc[nH]1 (2-hydroxypyridine → 2-pyridone)

Output DataFrame Columns:
    - orig_smiles: Original input SMILES (preserved for traceability)
    - smiles: Standardized molecule (with or without salts based on extract_salts)
    - salt: Removed salt/counterion as SMILES (only populated if extract_salts=True)

Salt Handling:
    Salt forms can dramatically affect properties like solubility.
    This module offers two modes for handling salts:

    When extract_salts=True (default, ChEMBL standard):
        - Removes salts/counterions to get parent molecule
        - Neutralizes charges on the parent
        - Records removed salts in 'salt' column
        Input: [Na+].CC(=O)[O-]  →  Parent: CC(=O)O, Salt: [Na+]

    When extract_salts=False (preserve full salt form):
        - Keeps all fragments including salts/counterions
        - Preserves ionic charges (no neutralization)

Mixture Detection:
    The module detects and logs potential mixtures (vs true salt forms):
    - Multiple large neutral organic fragments indicate a mixture
    - Mixtures are logged but NOT recorded in the salt column
    - True salts (small/charged fragments) are properly extracted

    Downstream modeling options:
    1. Use parent only (standard approach for most ADMET properties)
    2. Include salt as a categorical or computed feature
    3. Model parent + salt effects hierarchically
    4. Use full salt form for properties like solubility/formulation

References:
    - "ChEMBL Structure Pipeline" (Bento et al., 2020)
      https://doi.org/10.1186/s13321-020-00456-1
    - "Standardization and Validation with the RDKit" (Greg Landrum, RSC Open Science 2021)
      https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/Standardization%20and%20Validation%20with%20the%20RDKit.ipynb

Usage:
    from mol_standardize import standardize

    # Basic usage (removes salts by default, ChEMBL standard)
    df_std = standardize(df, smiles_column='smiles')

    # Keep salts in the molecule (preserve ionic forms)
    df_std = standardize(df, extract_salts=False)

    # Without tautomer canonicalization (faster, less aggressive)
    df_std = standardize(df, canonicalize_tautomer=False)
"""

import logging
from typing import Optional, Tuple
import pandas as pd
import time
from contextlib import contextmanager
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

log = logging.getLogger("workbench")
RDLogger.DisableLog("rdApp.warning")


# Helper context manager for timing
@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f"{name}: {time.time() - start:.2f}s")


class MolStandardizer:
    """
    Streamlined molecular standardizer for ADMET preprocessing
    Uses ChEMBL standardization pipeline with RDKit
    """

    def __init__(self, canonicalize_tautomer: bool = True, remove_salts: bool = True):
        """
        Initialize standardizer with ChEMBL defaults

        Args:
            canonicalize_tautomer: Whether to canonicalize tautomers (default True)
            remove_salts: Whether to remove salts/counterions (default True)
        """
        self.canonicalize_tautomer = canonicalize_tautomer
        self.remove_salts = remove_salts
        self.params = rdMolStandardize.CleanupParameters()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator(self.params)

    def standardize(self, mol: Mol) -> Tuple[Optional[Mol], Optional[str]]:
        """
        Main standardization pipeline for ADMET

        Pipeline:
        1. Cleanup (remove Hs, disconnect metals, normalize)
        2. Get largest fragment (optional - only if remove_salts=True)
           2a. Extract salt information BEFORE further modifications
        3. Neutralize charges
        4. Canonicalize tautomer (optional)

        Args:
            mol: RDKit molecule object

        Returns:
            Tuple of (standardized molecule or None if failed, salt SMILES or None)
        """
        if mol is None:
            return None, None

        try:
            # Step 1: Cleanup
            cleaned_mol = rdMolStandardize.Cleanup(mol, self.params)
            if cleaned_mol is None:
                return None, None

            # If not doing any transformations, return early
            if not self.remove_salts and not self.canonicalize_tautomer:
                return cleaned_mol, None

            salt_smiles = None
            mol = cleaned_mol

            # Step 2: Fragment handling (conditional based on remove_salts)
            if self.remove_salts:
                # Get parent molecule
                parent_mol = rdMolStandardize.FragmentParent(cleaned_mol, self.params)
                if parent_mol:
                    # Extract salt BEFORE any modifications to parent
                    salt_smiles = self._extract_salt(cleaned_mol, parent_mol)
                    mol = parent_mol
                else:
                    return None, None
            # If not removing salts, keep the full molecule intact

            # Step 3: Neutralize charges (skip if keeping salts to preserve ionic forms)
            if self.remove_salts:
                mol = rdMolStandardize.ChargeParent(mol, self.params, skipStandardize=True)
                if mol is None:
                    return None, salt_smiles

            # Step 4: Canonicalize tautomer (LAST STEP)
            if self.canonicalize_tautomer:
                mol = self.tautomer_enumerator.Canonicalize(mol)

            return mol, salt_smiles

        except Exception as e:
            log.warning(f"Standardization failed: {e}")
            return None, None

    def _extract_salt(self, orig_mol: Mol, parent_mol: Mol) -> Optional[str]:
        """
        Extract salt/counterion by comparing original and parent molecules.

        Detects and handles mixtures vs true salt forms:
        - True salts: small (<= 6 heavy atoms) or charged fragments
        - Mixtures: multiple large neutral organic fragments

        Args:
            orig_mol: Original molecule (after Cleanup, before FragmentParent)
            parent_mol: Parent molecule (after FragmentParent, before tautomerization)

        Returns:
            SMILES string of salt components or None if no salts/mixture detected
        """
        try:
            # Quick atom count check
            if orig_mol.GetNumAtoms() == parent_mol.GetNumAtoms():
                return None

            # Quick heavy atom difference check
            heavy_diff = orig_mol.GetNumHeavyAtoms() - parent_mol.GetNumHeavyAtoms()
            if heavy_diff <= 0:
                return None

            # Get all fragments from original molecule
            orig_frags = Chem.GetMolFrags(orig_mol, asMols=True)

            # If only one fragment, no salt
            if len(orig_frags) <= 1:
                return None

            # Get canonical SMILES of parent for comparison
            parent_smiles = Chem.MolToSmiles(parent_mol, canonical=True)

            # Separate fragments into salts vs potential mixture components
            salt_frags = []
            mixture_frags = []

            for frag in orig_frags:
                frag_smiles = Chem.MolToSmiles(frag, canonical=True)

                # Skip the parent fragment
                if frag_smiles == parent_smiles:
                    continue

                # Classify fragment as salt or mixture component
                num_heavy = frag.GetNumHeavyAtoms()
                has_charge = any(atom.GetFormalCharge() != 0 for atom in frag.GetAtoms())

                # More nuanced classification
                if has_charge and num_heavy <= 10:  # Small charged fragment - likely a salt
                    salt_frags.append(frag_smiles)
                elif not has_charge and num_heavy <= 6:  # Small neutral - could be solvent/salt
                    salt_frags.append(frag_smiles)
                else:
                    # Large neutral fragment - likely part of a mixture
                    mixture_frags.append(frag_smiles)

            # Check if this looks like a mixture
            if mixture_frags:
                # Log mixture detection
                total_frags = len(orig_frags)
                log.warning(
                    f"Mixture detected: {total_frags} total fragments, "
                    f"{len(mixture_frags)} large neutral organics. "
                    f"Removing: {'.'.join(mixture_frags + salt_frags)}"
                )
                # Return None for mixtures - don't pollute the salt column
                return None

            # Return actual salts only
            return ".".join(salt_frags) if salt_frags else None

        except Exception as e:
            log.info(f"Salt extraction failed: {e}")
            return None


def standardize(
    df: pd.DataFrame,
    canonicalize_tautomer: bool = True,
    extract_salts: bool = True,
) -> pd.DataFrame:
    """
    Standardize molecules in a DataFrame for ADMET modeling

    Args:
        df: Input DataFrame with SMILES column
        canonicalize_tautomer: Whether to canonicalize tautomers (default: True)
        extract_salts: Whether to remove and extract salts (default: True)
                      If False, keeps full molecule with salts/counterions intact,
                      skipping charge neutralization to preserve ionic character

    Returns:
        DataFrame with:
        - orig_smiles: Original SMILES (preserved)
        - smiles: Standardized SMILES (working column for downstream)
        - salt: Removed salt/counterion SMILES (only if extract_salts=True)
                None for mixtures or when no true salts present
    """

    # Check for the smiles column (any capitalization)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    # Copy input DataFrame to avoid modifying original
    result = df.copy()

    # Preserve original SMILES if not already saved
    if "orig_smiles" not in result.columns:
        result["orig_smiles"] = result[smiles_column]

    # Initialize standardizer
    standardizer = MolStandardizer(canonicalize_tautomer=canonicalize_tautomer, remove_salts=extract_salts)

    def process_smiles(smiles: str) -> pd.Series:
        """
        Process a single SMILES string through standardization pipeline

        Args:
            smiles: Input SMILES string

        Returns:
            Series with standardized SMILES and extracted salt (if applicable)
        """
        # Handle missing values
        if pd.isna(smiles) or smiles == "":
            log.error("Encountered missing or empty SMILES string")
            return pd.Series({"smiles": None, "salt": None})

        # Early check for unreasonably long SMILES
        if len(smiles) > 1000:
            log.error(f"SMILES too long ({len(smiles)} chars): {smiles[:50]}...")
            return pd.Series({"smiles": None, "salt": None})

        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.error(f"Invalid SMILES: {smiles}")
            return pd.Series({"smiles": None, "salt": None})

        # Full standardization with optional salt removal
        std_mol, salt_smiles = standardizer.standardize(mol)

        # After standardization, validate the result
        if std_mol is not None:
            # Check if molecule is reasonable
            if std_mol.GetNumAtoms() == 0 or std_mol.GetNumAtoms() > 200:  # Arbitrary limits
                log.error(f"Rejecting molecule size: {std_mol.GetNumAtoms()} atoms")
                log.error(f"Original SMILES: {smiles}")
                return pd.Series({"smiles": None, "salt": salt_smiles})

        if std_mol is None:
            return pd.Series(
                {
                    "smiles": None,
                    "salt": salt_smiles,  # May have extracted salt even if full standardization failed
                }
            )

        # Convert back to SMILES
        return pd.Series(
            {"smiles": Chem.MolToSmiles(std_mol, canonical=True), "salt": salt_smiles if extract_salts else None}
        )

    # Process molecules
    processed = result[smiles_column].apply(process_smiles)

    # Update the dataframe with processed results
    for col in ["smiles", "salt"]:
        result[col] = processed[col]

    return result


if __name__ == "__main__":

    # Pandas display options for better readability
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 100)

    # Test with DataFrame including various salt forms
    test_data = pd.DataFrame(
        {
            "smiles": [
                # Organic salts
                "[Na+].CC(=O)[O-]",  # Sodium acetate
                "CC(=O)O.CCN",  # Acetic acid + ethylamine (acid-base pair)
                # Tautomers
                "CC(=O)CC(C)=O",  # Acetylacetone - tautomer
                "c1ccc(O)nc1",  # 2-hydroxypyridine/2-pyridone - tautomer
                # Multi-fragment
                "CCO.CC",  # Ethanol + methane mixture
                # Simple organics
                "CC(C)(C)c1ccccc1",  # tert-butylbenzene
                # Carbonate salts
                "[Na+].[Na+].[O-]C([O-])=O",  # Sodium carbonate
                "[Li+].[Li+].[O-]C([O-])=O",  # Lithium carbonate
                "[K+].[K+].[O-]C([O-])=O",  # Potassium carbonate
                "[Mg++].[O-]C([O-])=O",  # Magnesium carbonate
                "[Ca++].[O-]C([O-])=O",  # Calcium carbonate
                # Drug salts
                "CC(C)NCC(O)c1ccc(O)c(O)c1.Cl",  # Isoproterenol HCl
                "CN1CCC[C@H]1c2cccnc2.[Cl-]",  # Nicotine HCl
                # Tautomer with salt
                "c1ccc(O)nc1.Cl",  # 2-hydroxypyridine with HCl
                # Edge cases
                None,  # Missing value
                "INVALID",  # Invalid SMILES
            ],
            "compound_id": [f"C{i:03d}" for i in range(1, 17)],
        }
    )

    # General test
    print("Testing standardization with full dataset...")
    standardize(test_data)

    # Remove the last two rows to avoid errors with None and INVALID
    test_data = test_data.iloc[:-2].reset_index(drop=True)

    # Test WITHOUT salt removal (keeps full molecule)
    print("\nStandardization KEEPING salts (extract_salts=False) Tautomerization: True")
    result_keep = standardize(test_data, extract_salts=False, canonicalize_tautomer=True)
    display_order = ["compound_id", "orig_smiles", "smiles", "salt"]
    print(result_keep[display_order])

    # Test WITH salt removal
    print("\n" + "=" * 70)
    print("Standardization REMOVING salts (extract_salts=True):")
    result_remove = standardize(test_data, extract_salts=True, canonicalize_tautomer=True)
    print(result_remove[display_order])

    # Test with problematic cases specifically
    print("\n" + "=" * 70)
    print("Testing specific problematic cases:")
    problem_cases = pd.DataFrame(
        {
            "smiles": [
                "CC(=O)O.CCN",  # Should extract CC(=O)O as salt
                "CCO.CC",  # Should return CC as salt
            ],
            "compound_id": ["TEST_C002", "TEST_C005"],
        }
    )

    problem_result = standardize(problem_cases, extract_salts=True, canonicalize_tautomer=True)
    print(problem_result[display_order])

    # Performance test with larger dataset
    from workbench.api import DataSource

    print("\n" + "=" * 70)

    ds = DataSource("aqsol_data")
    df = ds.pull_dataframe()[["id", "smiles"]][:1000]

    for tautomer in [True, False]:
        for extract in [True, False]:
            print(f"Performance test with AQSol dataset: tautomer={tautomer} extract_salts={extract}:")
            start_time = time.time()
            std_df = standardize(df, canonicalize_tautomer=tautomer, extract_salts=extract)
            elapsed = time.time() - start_time
            mol_per_sec = len(df) / elapsed
            print(f"{elapsed:.2f}s ({mol_per_sec:.0f} mol/s)")
