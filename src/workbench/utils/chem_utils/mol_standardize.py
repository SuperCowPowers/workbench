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
       
    2. Fragment Parent Selection
       - Identifies and keeps the largest organic fragment
       - Removes salts, solvents, and counterions
       - Example: [Na+].CC(=O)[O-] → CC(=O)O (keeps acetate, removes sodium)
       
    3. Charge Neutralization
       - Neutralizes charges where possible
       - Example: CC(=O)[O-] → CC(=O)O
       
    4. Tautomer Canonicalization (optional, default=True)
       - Generates canonical tautomer form for consistency
       - Example: Oc1ccccn1 → O=c1cccc[nH]1 (2-hydroxypyridine → 2-pyridone)

Output DataFrame Columns:
    - orig_smiles: Original input SMILES (preserved for traceability)
    - smiles: Standardized parent molecule (use for descriptor calculation)
    - salt: Removed salt/counterion as SMILES (e.g., "[Na+]", "[Cl-]", "Cl")
    - standardization_failed: Boolean flag indicating processing failures

Salt Handling:
    Salt forms can dramatically affect properties like solubility (up to 6 log units).
    This module preserves salt information to enable different modeling strategies:
    
    Example - Carbonates with different counterions:
        Input: [Na+].[Na+].[O-]C([O-])=O  →  Parent: O=C(O)O, Salt: [Na+].[Na+]
        Input: [Mg++].[O-]C([O-])=O       →  Parent: O=C(O)O, Salt: [Mg+2]
    
    Downstream modeling options:
    1. Use parent only (standard approach for most ADMET properties)
    2. Include salt as a categorical or computed feature
    3. Model parent + salt effects hierarchically

References:
    - "ChEMBL Structure Pipeline" (Bento et al., 2020)
      https://doi.org/10.1186/s13321-020-00456-1
    - "Standardization and Validation with the RDKit" (Greg Landrum, RSC Open Science 2021)
      https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/Standardization%20and%20Validation%20with%20the%20RDKit.ipynb

Usage:
    from mol_standardize import standardize
    
    # Basic usage
    df_std = standardize(df, smiles_column='smiles')
    
    # Without tautomer canonicalization (faster, less aggressive)
    df_std = standardize(df, canonicalize_tautomer=False)
    
    # Drop failed molecules
    df_std = standardize(df, drop_invalid=True)
"""

import logging
from typing import Optional, List, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


class MolStandardizer:
    """
    Streamlined molecular standardizer for ADMET preprocessing
    Uses ChEMBL standardization pipeline with RDKit
    """

    def __init__(self, canonicalize_tautomer: bool = True):
        """
        Initialize standardizer with ChEMBL defaults

        Args:
            canonicalize_tautomer: Whether to canonicalize tautomers (default True)
        """
        self.canonicalize_tautomer = canonicalize_tautomer
        self.params = rdMolStandardize.CleanupParameters()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator(self.params)

    def standardize(self, mol: Mol) -> Optional[Mol]:
        """
        Main standardization pipeline for ADMET

        Pipeline:
        1. Cleanup (remove Hs, disconnect metals, normalize)
        2. Get largest fragment
        3. Neutralize charges
        4. Canonicalize tautomer (optional)

        Args:
            mol: RDKit molecule object

        Returns:
            Standardized molecule or None if failed
        """
        if mol is None:
            return None

        try:
            # Step 1: Cleanup
            mol = rdMolStandardize.Cleanup(mol, self.params)
            if mol is None:
                return None

            # Step 2: Get largest fragment (removes salts, solvents)
            mol = rdMolStandardize.FragmentParent(mol, self.params)
            if mol is None:
                return None

            # Step 3: Neutralize charges
            mol = rdMolStandardize.ChargeParent(mol, self.params, skipStandardize=True)
            if mol is None:
                return None

            # Step 4: Canonicalize tautomer
            if self.canonicalize_tautomer:
                mol = self.tautomer_enumerator.Canonicalize(mol)

            return mol

        except Exception as e:
            logger.warning(f"Standardization failed: {e}")
            return None

    def extract_salt(self, orig_mol: Mol, parent_mol: Mol) -> Optional[str]:
        """
        Extract salt/counterion by comparing original and parent molecules

        Args:
            orig_mol: Original molecule (before FragmentParent)
            parent_mol: Parent molecule (after FragmentParent)

        Returns:
            SMILES string of salt components or None
        """
        if orig_mol is None or parent_mol is None:
            return None

        try:
            # Get all fragments from original molecule
            orig_frags = Chem.GetMolFrags(orig_mol, asMols=True)

            # If only one fragment, no salt
            if len(orig_frags) <= 1:
                return None

            # Get canonical SMILES of parent for comparison
            parent_smiles = Chem.MolToSmiles(parent_mol, canonical=True)

            # Collect non-parent fragments (salts/counterions)
            salt_frags = []
            for frag in orig_frags:
                frag_smiles = Chem.MolToSmiles(frag, canonical=True)
                if frag_smiles != parent_smiles:
                    salt_frags.append(frag_smiles)

            return '.'.join(salt_frags) if salt_frags else None

        except Exception as e:
            logger.debug(f"Salt extraction failed: {e}")
            return None


def standardize(df: pd.DataFrame,
                          smiles_column: str = 'smiles',
                          canonicalize_tautomer: bool = True,
                          extract_salts: bool = True,
                          drop_invalid: bool = False) -> pd.DataFrame:
    """
    Standardize molecules in a DataFrame for ADMET modeling

    Args:
        df: Input DataFrame with SMILES column
        smiles_column: Name of column containing SMILES (default: 'smiles')
        canonicalize_tautomer: Whether to canonicalize tautomers (default: True)
        extract_salts: Whether to extract salt information (default: True)
        drop_invalid: Whether to drop rows that fail standardization (default: False)

    Returns:
        DataFrame with:
        - orig_smiles: Original SMILES (preserved)
        - smiles: Standardized SMILES (working column for downstream)
        - salt: Removed salt/counterion SMILES (if extract_salts=True)
        - standardization_failed: Boolean flag for failures
    """
    result = df.copy()

    # Preserve original SMILES if not already saved
    if 'orig_smiles' not in result.columns:
        result['orig_smiles'] = result[smiles_column]

    # Initialize standardizer
    standardizer = MolStandardizer(canonicalize_tautomer=canonicalize_tautomer)

    def process_smiles(smiles):
        """Process a single SMILES string"""
        # Handle missing values
        if pd.isna(smiles) or smiles == '':
            return pd.Series({
                'smiles': None,
                'salt': None,
                'standardization_failed': True
            })

        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return pd.Series({
                'smiles': None,
                'salt': None,
                'standardization_failed': True
            })

        # Extract salt before standardization if requested
        salt_smiles = None
        if extract_salts:
            # Get the parent fragment for salt comparison
            # (this duplicates some work but keeps it simple)
            cleaned = rdMolStandardize.Cleanup(mol, standardizer.params)
            if cleaned:
                parent = rdMolStandardize.FragmentParent(cleaned, standardizer.params)
                if parent:
                    salt_smiles = standardizer.extract_salt(cleaned, parent)

        # Full standardization
        std_mol = standardizer.standardize(mol)
        if std_mol is None:
            return pd.Series({
                'smiles': None,
                'salt': salt_smiles,  # May have extracted salt even if full standardization failed
                'standardization_failed': True
            })

        # Convert back to SMILES
        std_smiles = Chem.MolToSmiles(std_mol, canonical=True)

        return pd.Series({
            'smiles': std_smiles,
            'salt': salt_smiles,
            'standardization_failed': False
        })

    # Process molecules
    processed = result[smiles_column].apply(process_smiles)

    # Update the dataframe with processed results
    result['smiles'] = processed['smiles']
    result['standardization_failed'] = processed['standardization_failed']

    if extract_salts:
        result['salt'] = processed['salt']

    # Calculate success rate
    success_count = (~result['standardization_failed']).sum()
    total_count = len(result)
    success_rate = success_count / total_count * 100

    logger.info(f"Standardization complete: {success_count}/{total_count} ({success_rate:.1f}%) successful")

    if extract_salts:
        salt_count = result['salt'].notna().sum()
        logger.info(f"Found {salt_count} molecules with salts/counterions")

    # Drop invalid if requested
    if drop_invalid:
        initial_count = len(result)
        result = result[~result['standardization_failed']].copy()
        dropped = initial_count - len(result)
        if dropped > 0:
            logger.info(f"Dropped {dropped} molecules that failed standardization")

    # Reorder columns for clarity
    cols = ['orig_smiles', 'smiles']
    if extract_salts:
        cols.append('salt')
    cols.append('standardization_failed')

    # Add remaining columns
    other_cols = [c for c in result.columns if c not in cols and c != smiles_column]
    result = result[cols + other_cols]

    return result


if __name__ == "__main__":
    # Test with DataFrame including various salt forms
    test_data = pd.DataFrame({
        'smiles': [
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

            # Edge cases
            None,  # Missing value
            "INVALID",  # Invalid SMILES
        ],
        'compound_id': [f'C{i:03d}' for i in range(1, 16)],
        'logS': [5.2, 4.1, 6.1, 7.3, 4.8, 5.5,
                 0.05, -0.76, 0.95, -2.18, -3.5,
                 3.2, 2.8, None, 6.0]
    })

    print("Testing molecular standardization pipeline with salt extraction")
    print("=" * 70)
    print("\nInput DataFrame (first 5 rows):")
    print(test_data.head())

    # Run standardization
    print("\n" + "=" * 70)
    print("Running standardization with salt extraction...")
    result_df = standardize(test_data, extract_salts=True)

    print("\n" + "=" * 70)
    print("Output DataFrame (selected columns):")
    display_cols = ['compound_id', 'orig_smiles', 'smiles', 'salt', 'logS', 'standardization_failed']
    print(result_df[display_cols].to_string())

    # Show specific examples
    print("\n" + "=" * 70)
    print("Salt extraction examples:")
    for idx, row in result_df.iterrows():
        if pd.notna(row['salt']):
            print(f"{row['compound_id']}: Parent={row['smiles'][:30]:30} Salt={row['salt']}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"Total molecules: {len(result_df)}")
    print(f"Successfully standardized: {(~result_df['standardization_failed']).sum()}")
    print(f"Molecules with salts: {result_df['salt'].notna().sum()}")
    print(f"Unique salts found: {result_df['salt'].dropna().unique()[:5]}...")  # First 5 unique salts