"""Compound Dataset Overlap Analysis

This module provides utilities for comparing two molecular datasets based on
Tanimoto similarity in fingerprint space. It helps quantify the "overlap"
between datasets in chemical space.

Use cases:
    - Train/test split validation: Ensure test set isn't too similar to training
    - Dataset comparison: Compare proprietary vs public datasets
    - Novelty assessment: Find compounds in dataset B that are novel vs dataset A
"""

import logging
from typing import Optional, Tuple

import pandas as pd

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

# Set up logging
log = logging.getLogger("workbench")


class CompoundDatasetOverlap:
    """Compare two molecular datasets using Tanimoto similarity.

    Combines datasets with a 'dataset' tag, uses FingerprintProximity to find
    neighbors, then filters to cross-dataset matches. For each compound in B,
    finds the most similar compound in A.

    Attributes:
        prox: FingerprintProximity instance on combined dataset
        overlap_df: Results DataFrame with similarity scores for each compound in B
    """

    def __init__(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        id_column_a: str = "id",
        id_column_b: str = "id",
        fingerprint_column: Optional[str] = None,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        """
        Initialize the CompoundDatasetOverlap analysis.

        Args:
            df_a: Reference dataset (DataFrame with SMILES or fingerprints)
            df_b: Query dataset (DataFrame with SMILES or fingerprints)
            id_column_a: ID column name in df_a
            id_column_b: ID column name in df_b
            fingerprint_column: Name of fingerprint column (if None, computes from SMILES)
            radius: Morgan fingerprint radius (default: 2 = ECFP4)
            n_bits: Number of fingerprint bits (default: 2048)
        """
        self.id_column_a = id_column_a
        self.id_column_b = id_column_b
        self._radius = radius
        self._n_bits = n_bits

        # Standardize ID column names and add dataset tags
        df_a = df_a.copy()
        df_b = df_b.copy()

        # Rename ID columns to common name if different
        if id_column_a != "id":
            df_a = df_a.rename(columns={id_column_a: "id"})
        if id_column_b != "id":
            df_b = df_b.rename(columns={id_column_b: "id"})

        df_a["dataset"] = "A"
        df_b["dataset"] = "B"

        # Combine datasets
        self.combined_df = pd.concat([df_a, df_b], ignore_index=True)
        log.info(f"Combined dataset: {len(df_a)} from A + {len(df_b)} from B = {len(self.combined_df)} total")

        # Build FingerprintProximity on combined data
        self.prox = FingerprintProximity(
            self.combined_df,
            id_column="id",
            fingerprint_column=fingerprint_column,
            radius=radius,
            n_bits=n_bits,
            include_all_columns=True,
        )

        # Compute cross-dataset overlap
        self.overlap_df = self._compute_cross_dataset_overlap()

    def _compute_cross_dataset_overlap(self) -> pd.DataFrame:
        """For each compound in B, find nearest neighbor in A."""
        # Get all compounds in B
        b_ids = self.combined_df[self.combined_df["dataset"] == "B"]["id"].tolist()
        log.info(f"Computing cross-dataset neighbors for {len(b_ids)} compounds in B")

        # Request enough neighbors to find cross-dataset matches (but not more than dataset size)
        n_neighbors = min(50, len(self.combined_df) - 1)

        results = []
        for b_id in b_ids:
            # Get neighbors (request more than needed to ensure we find cross-dataset match)
            neighbors = self.prox.neighbors(b_id, n_neighbors=n_neighbors, include_self=False)

            # Filter to dataset A only (neighbors are already sorted by similarity descending)
            cross_neighbors = neighbors[neighbors["dataset"] == "A"]

            if len(cross_neighbors) > 0:
                # Take the most similar (first row after sorting)
                best_match = cross_neighbors.iloc[0]
                results.append(
                    {
                        "id": b_id,
                        "nearest_neighbor_id": best_match["neighbor_id"],
                        "tanimoto_similarity": best_match["similarity"],
                    }
                )
            else:
                # No cross-dataset neighbor found (all neighbors were from B)
                results.append(
                    {
                        "id": b_id,
                        "nearest_neighbor_id": None,
                        "tanimoto_similarity": 0.0,
                    }
                )

        result_df = pd.DataFrame(results)

        # Add SMILES if available
        smiles_col = next((c for c in self.combined_df.columns if c.lower() == "smiles"), None)
        if smiles_col:
            b_smiles = self.combined_df[self.combined_df["dataset"] == "B"].set_index("id")[smiles_col]
            a_smiles = self.combined_df[self.combined_df["dataset"] == "A"].set_index("id")[smiles_col]
            result_df["smiles"] = result_df["id"].map(b_smiles)
            result_df["nearest_neighbor_smiles"] = result_df["nearest_neighbor_id"].map(a_smiles)

        return result_df.sort_values("tanimoto_similarity", ascending=False).reset_index(drop=True)

    def summary_stats(self) -> pd.DataFrame:
        """Return distribution statistics for nearest-neighbor Tanimoto similarities."""
        return (
            self.overlap_df["tanimoto_similarity"]
            .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            .to_frame()
        )

    def novel_compounds(self, threshold: float = 0.4) -> pd.DataFrame:
        """Return compounds in B that are novel (low similarity to A).

        Args:
            threshold: Maximum Tanimoto similarity to consider "novel" (default: 0.4)

        Returns:
            DataFrame of compounds in B with similarity below threshold
        """
        novel = self.overlap_df[self.overlap_df["tanimoto_similarity"] < threshold].copy()
        return novel.sort_values("tanimoto_similarity", ascending=True).reset_index(drop=True)

    def similar_compounds(self, threshold: float = 0.7) -> pd.DataFrame:
        """Return compounds in B that are similar to A (high overlap).

        Args:
            threshold: Minimum Tanimoto similarity to consider "similar" (default: 0.7)

        Returns:
            DataFrame of compounds in B with similarity above threshold
        """
        similar = self.overlap_df[self.overlap_df["tanimoto_similarity"] >= threshold].copy()
        return similar.sort_values("tanimoto_similarity", ascending=False).reset_index(drop=True)

    def overlap_fraction(self, threshold: float = 0.7) -> float:
        """Return fraction of B that overlaps with A above similarity threshold.

        Args:
            threshold: Minimum Tanimoto similarity to consider "overlapping"

        Returns:
            Fraction of compounds in B with nearest neighbor similarity >= threshold
        """
        n_overlapping = (self.overlap_df["tanimoto_similarity"] >= threshold).sum()
        return n_overlapping / len(self.overlap_df)

    def plot_histogram(self, bins: int = 50, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot histogram of nearest-neighbor Tanimoto similarities.

        Args:
            bins: Number of histogram bins
            figsize: Figure size (width, height)
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self.overlap_df["tanimoto_similarity"], bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Tanimoto Similarity (B → nearest in A)")
        ax.set_ylabel("Count")
        ax.set_title(f"Dataset Overlap: {len(self.overlap_df)} compounds in B")
        ax.axvline(x=0.4, color="red", linestyle="--", label="Novel threshold (0.4)")
        ax.axvline(x=0.7, color="green", linestyle="--", label="Similar threshold (0.7)")
        ax.legend()

        # Add summary stats as text
        stats = self.overlap_df["tanimoto_similarity"]
        textstr = f"Mean: {stats.mean():.3f}\nMedian: {stats.median():.3f}\nStd: {stats.std():.3f}"
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.show()


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing CompoundDatasetOverlap")
    print("=" * 80)

    # Test 1: Basic functionality with synthetic data
    print("\n1. Testing with synthetic fingerprint data...")

    # Create synthetic datasets with known overlap (16-bit fingerprints for clarity)
    # Dataset A: Reference compounds
    data_a = {
        "id": ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
        "fingerprint": [
            "1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0",  # Pattern 1: alternating starting with 1
            "0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1",  # Pattern 2: alternating starting with 0
            "1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0",  # Pattern 3: blocks of 4
            "0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1",  # Pattern 4: inverse of 3
            "1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0",  # Pattern 5: half 1s, half 0s
            "0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1",  # Pattern 6: inverse of 5
            "1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1",  # Pattern 7: pairs
            "1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0",  # Pattern 8: double alternating
        ],
    }

    # Dataset B: Query compounds with known relationships to A
    data_b = {
        "id": ["b1", "b2", "b3", "b4", "b5", "b6"],
        "fingerprint": [
            "1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0",  # Exact match to a1 (sim=1.0)
            "1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0",  # 1 bit different from a1 (sim~0.875)
            "0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1",  # Exact match to a2 (sim=1.0)
            "1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0",  # Similar to a3 (sim~0.7)
            "0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0",  # Very different from all (low sim)
            "1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0",  # Exact match to a8 (sim=1.0)
        ],
    }

    df_a = pd.DataFrame(data_a)
    df_b = pd.DataFrame(data_b)

    print(f"   Dataset A: {len(df_a)} compounds, Dataset B: {len(df_b)} compounds")

    overlap = CompoundDatasetOverlap(df_a, df_b, id_column_a="id", id_column_b="id")

    print("\n   Overlap results (expected: b1→a1, b2→a1, b3→a2, b4→a3, b5→low sim, b6→a8):")
    print(overlap.overlap_df.to_string(index=False))

    print("\n   Summary statistics:")
    print(overlap.summary_stats())

    print(f"\n   Overlap fraction (threshold=0.7): {overlap.overlap_fraction(0.7):.2%}")
    print(f"   Novel compounds (threshold=0.4): {len(overlap.novel_compounds(0.4))}")

    # Test 2: With real SMILES data
    print("\n2. Testing with SMILES data (computing fingerprints)...")

    # Dataset A: Reference drug-like compounds
    smiles_a = {
        "id": ["aspirin", "caffeine", "glucose", "ibuprofen", "naproxen", "ethanol", "methanol", "propanol"],
        "smiles": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
            "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",  # glucose
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
            "COC1=CC2=CC(C(C)C(O)=O)=CC=C2C=C1",  # naproxen
            "CCO",  # ethanol
            "CO",  # methanol
            "CCCO",  # propanol
        ],
    }

    # Dataset B: Query compounds with various similarities
    smiles_b = {
        "id": ["acetaminophen", "theophylline", "benzene", "toluene", "phenol", "aniline"],
        "smiles": [
            "CC(=O)NC1=CC=C(C=C1)O",  # acetaminophen - similar to aspirin
            "CN1C=NC2=C1C(=O)NC(=O)N2",  # theophylline - similar to caffeine
            "c1ccccc1",  # benzene - simple aromatic
            "Cc1ccccc1",  # toluene - similar to benzene
            "Oc1ccccc1",  # phenol - hydroxyl benzene
            "Nc1ccccc1",  # aniline - amino benzene
        ],
    }

    df_a_smiles = pd.DataFrame(smiles_a)
    df_b_smiles = pd.DataFrame(smiles_b)

    print(f"   Dataset A: {len(df_a_smiles)} compounds, Dataset B: {len(df_b_smiles)} compounds")

    overlap_smiles = CompoundDatasetOverlap(
        df_a_smiles, df_b_smiles, id_column_a="id", id_column_b="id", radius=2, n_bits=1024
    )

    print("\n   Overlap results:")
    print(overlap_smiles.overlap_df[["id", "nearest_neighbor_id", "tanimoto_similarity"]].to_string(index=False))

    print("\n   Summary statistics:")
    print(overlap_smiles.summary_stats())

    # Test 3: Novel and similar compound identification
    print("\n3. Testing novel/similar compound identification...")

    similar = overlap_smiles.similar_compounds(threshold=0.3)
    print(f"   Similar compounds (sim >= 0.3): {len(similar)}")
    if len(similar) > 0:
        print(similar[["id", "nearest_neighbor_id", "tanimoto_similarity"]].to_string(index=False))

    novel = overlap_smiles.novel_compounds(threshold=0.3)
    print(f"\n   Novel compounds (sim < 0.3): {len(novel)}")
    if len(novel) > 0:
        print(novel[["id", "nearest_neighbor_id", "tanimoto_similarity"]].to_string(index=False))

    # Test 4: With Workbench data (if available)
    print("\n4. Testing with Workbench FeatureSet (if available)...")

    try:
        from workbench.api import FeatureSet

        fs = FeatureSet("aqsol_features")
        full_df = fs.pull_dataframe()[:1000]  # Limit to first 1000 for testing

        # Split into two sets
        df_train = full_df.sample(frac=0.8, random_state=42)
        df_test = full_df.drop(df_train.index)

        print(f"   Train set: {len(df_train)} compounds")
        print(f"   Test set: {len(df_test)} compounds")

        overlap_real = CompoundDatasetOverlap(df_train, df_test, id_column_a=fs.id_column, id_column_b=fs.id_column)

        print("\n   Summary statistics:")
        print(overlap_real.summary_stats())

        print(f"\n   Overlap fraction (sim >= 0.7): {overlap_real.overlap_fraction(0.7):.2%}")
        print(f"   Overlap fraction (sim >= 0.5): {overlap_real.overlap_fraction(0.5):.2%}")
        print(f"   Novel compounds (sim < 0.4): {len(overlap_real.novel_compounds(0.4))}")

        # Uncomment to show histogram
        # overlap_real.plot_histogram()

    except Exception as e:
        print(f"   Skipping Workbench test: {e}")

    print("\n" + "=" * 80)
    print("✅ All CompoundDatasetOverlap tests completed!")
    print("=" * 80)
