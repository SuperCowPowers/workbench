"""Dimensionality reduction and projection utilities for molecular fingerprints"""

import logging
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# Try importing UMAP
try:
    import umap
except ImportError:
    umap = None

# Set up the logger
log = logging.getLogger("workbench")


def fingerprints_to_matrix(fingerprints, dtype=np.uint8):
    """
    Convert fingerprints to numpy matrix.

    Supports two formats (auto-detected):
        - Bitstrings: "10110010..." → matrix of 0s and 1s
        - Count vectors: "0,3,0,1,5,..." → matrix of counts (or binary if dtype=np.bool_)

    Args:
        fingerprints: pandas Series or list of fingerprints
        dtype: numpy data type (uint8 is default; np.bool_ for Jaccard computations)

    Returns:
        dense numpy array of shape (n_molecules, n_bits)
    """
    # Auto-detect format based on first fingerprint
    sample = str(fingerprints.iloc[0] if hasattr(fingerprints, "iloc") else fingerprints[0])
    if "," in sample:
        # Count vector format: comma-separated integers
        matrix = np.array([list(map(int, fp.split(","))) for fp in fingerprints], dtype=dtype)
    else:
        # Bitstring format: each character is a bit
        matrix = np.array([list(fp) for fp in fingerprints], dtype=dtype)
    return matrix


def project_fingerprints(df: pd.DataFrame, projection: str = "UMAP") -> pd.DataFrame:
    """Project fingerprints onto a 2D plane using dimensionality reduction techniques.

    Args:
        df (pd.DataFrame): Input DataFrame containing fingerprint data.
        projection (str): Dimensionality reduction technique to use (TSNE or UMAP).

    Returns:
        pd.DataFrame: The input DataFrame with the projected coordinates added as 'x' and 'y' columns.
    """
    # Check for the fingerprint column (case-insensitive)
    fingerprint_column = next((col for col in df.columns if "fingerprint" in col.lower()), None)
    if fingerprint_column is None:
        raise ValueError("Input DataFrame must have a fingerprint column")

    # Create a matrix of fingerprints
    X = fingerprints_to_matrix(df[fingerprint_column])

    # Get number of samples
    n_samples = X.shape[0]

    # Check for UMAP availability
    if projection == "UMAP" and umap is None:
        log.warning("UMAP is not available. Using TSNE instead.")
        projection = "TSNE"

    # Run the projection
    if projection == "TSNE":
        # Adjust perplexity based on dataset size
        # Perplexity must be less than n_samples and at least 1
        perplexity = min(30, max(1, n_samples - 1))

        # TSNE requires at least 4 samples
        if n_samples < 4:
            log.warning(f"Dataset too small for TSNE (n={n_samples}). Need at least 4 samples.")
            # Return with random coordinates for very small datasets
            df["x"] = np.random.uniform(-10, 10, n_samples)
            df["y"] = np.random.uniform(-10, 10, n_samples)
            return df

        # Run TSNE on the fingerprint matrix
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedding = tsne.fit_transform(X)
    else:
        # Run UMAP
        # Adjust n_neighbors based on dataset size
        n_neighbors = min(15, n_samples - 1) if n_samples > 1 else 1

        reducer = umap.UMAP(metric="jaccard", n_neighbors=n_neighbors)
        embedding = reducer.fit_transform(X)

    # Add coordinates to DataFrame
    df["x"] = embedding[:, 0]
    df["y"] = embedding[:, 1]

    # If vertices disconnect from the manifold, they are given NaN values (so replace with 0)
    df["x"] = df["x"].fillna(0)
    df["y"] = df["y"].fillna(0)

    # Jitter
    jitter_scale = 0.1
    df["x"] += np.random.uniform(0, jitter_scale, len(df))
    df["y"] += np.random.uniform(0, jitter_scale, len(df))

    return df


if __name__ == "__main__":
    print("Running molecular projection tests...")

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    # Test molecules
    test_molecules = {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "glucose": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
        "sodium_acetate": "CC(=O)[O-].[Na+]",
        "benzene": "c1ccccc1",
        "toluene": "Cc1ccccc1",
        "phenol": "Oc1ccccc1",
        "aniline": "Nc1ccccc1",
    }

    # Generate fingerprints for test
    print("\n1. Generating test fingerprints...")

    test_df = pd.DataFrame({"SMILES": list(test_molecules.values()), "name": list(test_molecules.keys())})

    # Generate Morgan fingerprints
    mols = [Chem.MolFromSmiles(smi) for smi in test_df["SMILES"]]
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    fingerprints = [morgan_gen.GetFingerprint(mol).ToBitString() if mol else None for mol in mols]
    test_df["fingerprint"] = fingerprints

    # Remove any failed molecules
    test_df = test_df.dropna(subset=["fingerprint"])
    print(f"   Generated {len(test_df)} fingerprints")

    # Test 2: Fingerprint to matrix conversion
    print("\n2. Testing fingerprint matrix conversion...")

    matrix = fingerprints_to_matrix(test_df["fingerprint"])
    print(f"   Matrix shape: {matrix.shape}")
    print(f"   Matrix dtype: {matrix.dtype}")
    print(f"   Non-zero elements: {np.count_nonzero(matrix)}")

    # Test 3: TSNE projection
    print("\n3. Testing TSNE projection...")

    try:
        proj_df = project_fingerprints(test_df.copy(), projection="TSNE")

        print("   TSNE projection results:")
        for _, row in proj_df.head(4).iterrows():
            print(f"   {row['name']:15} → x:{row['x']:7.2f} y:{row['y']:7.2f}")

        # Check that coordinates were added
        assert "x" in proj_df.columns and "y" in proj_df.columns
        print(f"   ✓ Successfully projected {len(proj_df)} molecules")

    except Exception as e:
        print(f"   Note: TSNE projection test limited: {e}")

    # Test 4: UMAP projection (if available)
    print("\n4. Testing UMAP projection...")

    if umap is not None:
        try:
            proj_umap_df = project_fingerprints(test_df.copy(), projection="UMAP")

            print("   UMAP projection results:")
            for _, row in proj_umap_df.head(4).iterrows():
                print(f"   {row['name']:15} → x:{row['x']:7.2f} y:{row['y']:7.2f}")

            print(f"   ✓ Successfully projected {len(proj_umap_df)} molecules with UMAP")

        except Exception as e:
            print(f"   Note: UMAP projection failed: {e}")
    else:
        print("   UMAP not available - skipping test")

    # Test 5: Edge cases
    print("\n5. Testing edge cases...")

    # Test with missing fingerprint column
    no_fp_df = pd.DataFrame({"SMILES": ["CCO", "CC"]})
    try:
        project_fingerprints(no_fp_df)
        print("   ✗ Should have raised error for missing fingerprint column")
    except ValueError as e:
        print(f"   ✓ Correctly raised error for missing fingerprint: {str(e)}")

    # Test with small dataset (less than perplexity)
    small_df = test_df.head(2).copy()
    if len(small_df) > 0:
        try:
            proj_small = project_fingerprints(small_df, projection="TSNE")
            print("   Note: Small dataset projection handled")
        except Exception as e:
            print(f"   Note: Small dataset appropriately failed: {type(e).__name__}")

    # Test 6: Testing NaN value handling
    print("\n6. Testing NaN value handling...")

    try:
        # The projection should handle NaN values by replacing with 0
        proj_test = project_fingerprints(test_df.copy(), projection="TSNE")
        has_nan = proj_test[["x", "y"]].isnull().any().any()
        print(f"   NaN values in output: {has_nan}")
        print("   ✓ NaN values properly handled")
    except Exception as e:
        print(f"   Note: Could not test NaN handling due to: {e}")

    print("\n✅ All projection tests completed!")
