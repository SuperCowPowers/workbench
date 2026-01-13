"""Compound Dataset Overlap Analysis

This module provides utilities for comparing two molecular datasets based on
Tanimoto similarity in fingerprint space. It helps quantify the "overlap"
between datasets in chemical space.

Use cases:
    - Train/test split validation: Ensure test set isn't too similar to training
    - Dataset comparison: Compare proprietary vs public datasets
    - Novelty assessment: Find compounds in query dataset that are novel vs reference
"""

import logging
from typing import Optional, Tuple

import pandas as pd

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

# Set up logging
log = logging.getLogger("workbench")


class CompoundDatasetOverlap:
    """Compare two molecular datasets using Tanimoto similarity.

    Builds a FingerprintProximity model on the reference dataset, then queries
    with SMILES from the query dataset to find the nearest neighbor in the
    reference for each query compound. This guarantees cross-dataset matches.

    Attributes:
        prox: FingerprintProximity instance on reference dataset
        overlap_df: Results DataFrame with similarity scores for each query compound
    """

    def __init__(
        self,
        df_reference: pd.DataFrame,
        df_query: pd.DataFrame,
        id_column_reference: str = "id",
        id_column_query: str = "id",
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        """
        Initialize the CompoundDatasetOverlap analysis.

        Args:
            df_reference: Reference dataset (DataFrame with SMILES)
            df_query: Query dataset (DataFrame with SMILES)
            id_column_reference: ID column name in df_reference
            id_column_query: ID column name in df_query
            radius: Morgan fingerprint radius (default: 2 = ECFP4)
            n_bits: Number of fingerprint bits (default: 2048)
        """
        self.id_column_reference = id_column_reference
        self.id_column_query = id_column_query
        self._radius = radius
        self._n_bits = n_bits

        # Store copies of the dataframes
        self.df_reference = df_reference.copy()
        self.df_query = df_query.copy()

        # Find SMILES columns
        self._smiles_col_reference = self._find_smiles_column(self.df_reference)
        self._smiles_col_query = self._find_smiles_column(self.df_query)

        if self._smiles_col_reference is None:
            raise ValueError("Reference dataset must have a SMILES column")
        if self._smiles_col_query is None:
            raise ValueError("Query dataset must have a SMILES column")

        log.info(f"Reference dataset: {len(self.df_reference)} compounds")
        log.info(f"Query dataset: {len(self.df_query)} compounds")

        # Build FingerprintProximity on reference dataset only
        self.prox = FingerprintProximity(
            self.df_reference,
            id_column=id_column_reference,
            radius=radius,
            n_bits=n_bits,
        )

        # Compute cross-dataset overlap
        self.overlap_df = self._compute_cross_dataset_overlap()

    @staticmethod
    def _find_smiles_column(df: pd.DataFrame) -> Optional[str]:
        """Find the SMILES column in a DataFrame (case-insensitive)."""
        for col in df.columns:
            if col.lower() == "smiles":
                return col
        return None

    def _compute_cross_dataset_overlap(self) -> pd.DataFrame:
        """For each query compound, find nearest neighbor in reference using neighbors_from_smiles."""
        log.info(f"Computing nearest neighbors in reference for {len(self.df_query)} query compounds")

        # Get SMILES list from query dataset
        query_smiles = self.df_query[self._smiles_col_query].tolist()
        query_ids = self.df_query[self.id_column_query].tolist()

        # Query all compounds against reference (get only nearest neighbor)
        neighbors_df = self.prox.neighbors_from_smiles(query_smiles, n_neighbors=1)

        # Build results with query IDs
        results = []
        for i, (q_id, q_smi) in enumerate(zip(query_ids, query_smiles)):
            # Find the row for this query SMILES
            match = neighbors_df[neighbors_df["query_id"] == q_smi]
            if len(match) > 0:
                row = match.iloc[0]
                results.append(
                    {
                        "id": q_id,
                        "smiles": q_smi,
                        "nearest_neighbor_id": row["neighbor_id"],
                        "tanimoto_similarity": row["similarity"],
                    }
                )
            else:
                # Should not happen, but handle gracefully
                results.append(
                    {
                        "id": q_id,
                        "smiles": q_smi,
                        "nearest_neighbor_id": None,
                        "tanimoto_similarity": 0.0,
                    }
                )

        result_df = pd.DataFrame(results)

        # Add nearest neighbor SMILES from reference
        ref_smiles_map = self.df_reference.set_index(self.id_column_reference)[self._smiles_col_reference]
        result_df["nearest_neighbor_smiles"] = result_df["nearest_neighbor_id"].map(ref_smiles_map)

        return result_df.sort_values("tanimoto_similarity", ascending=False).reset_index(drop=True)

    def summary_stats(self) -> pd.DataFrame:
        """Return distribution statistics for nearest-neighbor Tanimoto similarities."""
        return (
            self.overlap_df["tanimoto_similarity"]
            .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            .to_frame()
        )

    def novel_compounds(self, threshold: float = 0.4) -> pd.DataFrame:
        """Return query compounds that are novel (low similarity to reference).

        Args:
            threshold: Maximum Tanimoto similarity to consider "novel" (default: 0.4)

        Returns:
            DataFrame of query compounds with similarity below threshold
        """
        novel = self.overlap_df[self.overlap_df["tanimoto_similarity"] < threshold].copy()
        return novel.sort_values("tanimoto_similarity", ascending=True).reset_index(drop=True)

    def similar_compounds(self, threshold: float = 0.7) -> pd.DataFrame:
        """Return query compounds that are similar to reference (high overlap).

        Args:
            threshold: Minimum Tanimoto similarity to consider "similar" (default: 0.7)

        Returns:
            DataFrame of query compounds with similarity above threshold
        """
        similar = self.overlap_df[self.overlap_df["tanimoto_similarity"] >= threshold].copy()
        return similar.sort_values("tanimoto_similarity", ascending=False).reset_index(drop=True)

    def overlap_fraction(self, threshold: float = 0.7) -> float:
        """Return fraction of query compounds that overlap with reference above similarity threshold.

        Args:
            threshold: Minimum Tanimoto similarity to consider "overlapping"

        Returns:
            Fraction of query compounds with nearest neighbor similarity >= threshold
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
        ax.set_xlabel("Tanimoto Similarity (query → nearest in reference)")
        ax.set_ylabel("Count")
        ax.set_title(f"Dataset Overlap: {len(self.overlap_df)} query compounds")
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

    # Test 1: Basic functionality with SMILES data
    print("\n1. Testing with SMILES data...")

    # Reference dataset: Known drug-like compounds
    reference_data = {
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

    # Query dataset: Compounds to compare against reference
    query_data = {
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

    df_reference = pd.DataFrame(reference_data)
    df_query = pd.DataFrame(query_data)

    print(f"   Reference: {len(df_reference)} compounds, Query: {len(df_query)} compounds")

    overlap = CompoundDatasetOverlap(
        df_reference, df_query, id_column_reference="id", id_column_query="id", radius=2, n_bits=1024
    )

    print("\n   Overlap results:")
    print(overlap.overlap_df[["id", "nearest_neighbor_id", "tanimoto_similarity"]].to_string(index=False))

    print("\n   Summary statistics:")
    print(overlap.summary_stats())

    # Test 2: Novel and similar compound identification
    print("\n2. Testing novel/similar compound identification...")

    similar = overlap.similar_compounds(threshold=0.3)
    print(f"   Similar compounds (sim >= 0.3): {len(similar)}")
    if len(similar) > 0:
        print(similar[["id", "nearest_neighbor_id", "tanimoto_similarity"]].to_string(index=False))

    novel = overlap.novel_compounds(threshold=0.3)
    print(f"\n   Novel compounds (sim < 0.3): {len(novel)}")
    if len(novel) > 0:
        print(novel[["id", "nearest_neighbor_id", "tanimoto_similarity"]].to_string(index=False))

    # Test 3: With Workbench data (if available)
    print("\n3. Testing with Workbench FeatureSet (if available)...")

    try:
        from workbench.api import FeatureSet

        fs = FeatureSet("aqsol_features")
        full_df = fs.pull_dataframe()[:1000]  # Limit to first 1000 for testing

        # Split into reference and query sets
        df_reference = full_df.sample(frac=0.8, random_state=42)
        df_query = full_df.drop(df_reference.index)

        print(f"   Reference set: {len(df_reference)} compounds")
        print(f"   Query set: {len(df_query)} compounds")

        overlap = CompoundDatasetOverlap(
            df_reference, df_query, id_column_reference=fs.id_column, id_column_query=fs.id_column
        )

        print("\n   Summary statistics:")
        print(overlap.summary_stats())

        print(f"\n   Overlap fraction (sim >= 0.7): {overlap.overlap_fraction(0.7):.2%}")
        print(f"   Overlap fraction (sim >= 0.5): {overlap.overlap_fraction(0.5):.2%}")
        print(f"   Novel compounds (sim < 0.4): {len(overlap.novel_compounds(0.4))}")

        # Uncomment to show histogram
        overlap.plot_histogram()

    except Exception as e:
        print(f"   Skipping Workbench test: {e}")

    print("\n" + "=" * 80)
    print("✅ All CompoundDatasetOverlap tests completed!")
    print("=" * 80)
