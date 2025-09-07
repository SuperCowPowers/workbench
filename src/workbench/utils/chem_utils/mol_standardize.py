"""
mol_standardize.py - Molecular Standardization for ADMET Preprocessing
Following ChEMBL structure standardization pipeline

References:
    - "ChEMBL Structure Pipeline" (Bento et al., 2020)
      https://doi.org/10.1186/s13321-020-00456-1
    - "Standardization and Validation with the RDKit" (Greg Landrum, RSC Open Science 2021)
      https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/Standardization%20and%20Validation%20with%20the%20RDKit.ipynb
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

        # Use ChEMBL default parameters
        self.params = rdMolStandardize.CleanupParameters()

        # Initialize components
        self.uncharger = rdMolStandardize.Uncharger()
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
            # Step 1: Cleanup - removes Hs, disconnects metals, normalizes
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

            # Step 4: Canonicalize tautomer (optional but recommended for ADMET)
            if self.canonicalize_tautomer:
                mol = self.tautomer_enumerator.Canonicalize(mol)

            return mol

        except Exception as e:
            logger.warning(f"Standardization failed: {e}")
            return None


def validate_mol(mol: Mol) -> Tuple[bool, List[str]]:
    """
    Basic validation for ADMET preprocessing

    Args:
        mol: RDKit molecule

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    if mol is None:
        return False, ["Molecule is None"]

    issues = []

    # Check for multiple fragments (should be cleaned by standardization)
    if len(Chem.GetMolFrags(mol)) > 1:
        issues.append("Multiple fragments detected")

    # Check sanitization
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        issues.append(f"Sanitization failed: {str(e)}")

    # Check molecular weight (typical ADMET range)
    from rdkit.Chem.Descriptors import ExactMolWt
    mw = ExactMolWt(mol)
    if mw > 1000:
        issues.append(f"High molecular weight: {mw:.2f}")
    elif mw < 50:
        issues.append(f"Low molecular weight: {mw:.2f}")

    return len(issues) == 0, issues


def standardize_dataframe(df: pd.DataFrame,
                          smiles_column: str = 'smiles',
                          canonicalize_tautomer: bool = True,
                          validate: bool = True,
                          drop_invalid: bool = False) -> pd.DataFrame:
    """
    Standardize molecules in a DataFrame for ADMET modeling

    Standard workflow:
    - Input: DataFrame with 'smiles' column
    - Output: DataFrame with 'orig_smiles', 'smiles' (standardized), and QC columns

    Args:
        df: Input DataFrame with SMILES column
        smiles_column: Name of column containing SMILES (default: 'smiles')
        canonicalize_tautomer: Whether to canonicalize tautomers (default: True)
        validate: Whether to add validation columns (default: True)
        drop_invalid: Whether to drop rows that fail standardization (default: False)

    Returns:
        DataFrame with:
        - orig_smiles: Original SMILES (preserved)
        - smiles: Standardized SMILES (working column for downstream)
        - standardization_failed: Boolean flag for failures
        - validation_issues: List of any validation issues (if validate=True)
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
                'standardization_failed': True,
                'validation_issues': ['Missing SMILES']
            })

        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return pd.Series({
                'smiles': None,
                'standardization_failed': True,
                'validation_issues': ['Invalid SMILES']
            })

        # Standardize
        std_mol = standardizer.standardize(mol)
        if std_mol is None:
            return pd.Series({
                'smiles': None,
                'standardization_failed': True,
                'validation_issues': ['Standardization failed']
            })

        # Convert back to SMILES
        std_smiles = Chem.MolToSmiles(std_mol, canonical=True)

        # Validate if requested
        validation_issues = []
        if validate:
            is_valid, issues = validate_mol(std_mol)
            validation_issues = issues

        return pd.Series({
            'smiles': std_smiles,
            'standardization_failed': False,
            'validation_issues': validation_issues
        })

    # Process molecules (simple serial processing)
    processed = result[smiles_column].apply(process_smiles)

    # Update the dataframe with processed results
    result['smiles'] = processed['smiles']
    result['standardization_failed'] = processed['standardization_failed']

    if validate:
        result['validation_issues'] = processed['validation_issues']

    # Calculate success rate
    success_count = (~result['standardization_failed']).sum()
    total_count = len(result)
    success_rate = success_count / total_count * 100

    logger.info(f"Standardization complete: {success_count}/{total_count} ({success_rate:.1f}%) successful")

    if validate:
        # Report validation issues
        has_issues = result['validation_issues'].apply(lambda x: len(x) > 0)
        if has_issues.any():
            issue_count = has_issues.sum()
            logger.info(f"Validation issues found in {issue_count} molecules")

    # Drop invalid if requested
    if drop_invalid:
        initial_count = len(result)
        result = result[~result['standardization_failed']].copy()
        dropped = initial_count - len(result)
        if dropped > 0:
            logger.info(f"Dropped {dropped} molecules that failed standardization")

    # Reorder columns for clarity (keep orig_smiles and smiles at front)
    cols = ['orig_smiles', 'smiles', 'standardization_failed']
    if validate:
        cols.append('validation_issues')

    # Add remaining columns
    other_cols = [c for c in result.columns if c not in cols and c != smiles_column]
    result = result[cols + other_cols]

    return result


if __name__ == "__main__":
    # Test with DataFrame
    test_data = pd.DataFrame({
        'smiles': [
            "[Na+].CC(=O)[O-]",  # Sodium acetate - will be neutralized
            "CC(=O)CC(C)=O",  # Acetylacetone - tautomer
            "c1ccc(O)nc1",  # 2-hydroxypyridine/2-pyridone - tautomer
            "CCO.CC",  # Multi-fragment - will keep largest
            "CC(C)(C)c1ccccc1",  # tert-butylbenzene - should be fine
            None,  # Missing value
            "INVALID",  # Invalid SMILES
        ],
        'compound_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007'],
        'activity': [5.2, 6.1, 7.3, 4.8, 5.5, None, 6.0]
    })

    print("Testing molecular standardization pipeline")
    print("=" * 60)
    print("\nInput DataFrame:")
    print(test_data)

    # Run standardization
    print("\n" + "=" * 60)
    print("Running standardization...")
    result_df = standardize_dataframe(test_data, validate=True)

    print("\n" + "=" * 60)
    print("Output DataFrame:")
    print(result_df)

    # Show specific examples
    print("\n" + "=" * 60)
    print("Standardization examples:")
    for idx, row in result_df.iterrows():
        if pd.notna(row['orig_smiles']):
            print(f"{row['compound_id']}: {row['orig_smiles'][:30]:30} → {row['smiles']}")
            if row['validation_issues']:
                print(f"     Issues: {row['validation_issues']}")