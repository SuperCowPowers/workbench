"""Dataset Concordance: Compare two molecular datasets for chemical space overlap and SAR concordance.

Takes reference and query DataFrames, combines them with a ``dataset`` bookkeeping
column, and builds one FingerprintProximity model on the combined data. This gives
a shared UMAP projection and a single NN model that can be queried for cross-dataset
neighbors by filtering on the ``dataset`` column.

Produces:
    - **concordance_df**: Per-query-compound overlap (Tanimoto) and SAR concordance (residual).

This DataFrame, plus the shared 2D coordinates in ``self._prox.df``, back the
ConcordanceMap visualization UI.

Use cases:
    - Data fusion: Can proprietary and public ADMET data be safely merged for training?
    - Assay concordance: Do two assays measuring the same endpoint agree?
    - Model monitoring: Has the target relationship drifted in new data?

References:
    - Landrum & Riniker (2024) "Combining IC50 or Ki Values from Different Sources
      Is a Source of Significant Noise" JCIM
    - Parrondo-Pizarro et al. (2025) "Enhancing molecular property prediction through
      data integration and consistency assessment" J. Cheminform.
"""

import logging
import pandas as pd

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

# Set up logging
log = logging.getLogger("workbench")


class DatasetConcordance:
    """Compare two molecular datasets for chemical space overlap and SAR concordance.

    Takes reference and query DataFrames, combines them internally, and builds a single
    FingerprintProximity model (shared fingerprints, NN model, and UMAP projection).
    For each query compound, finds reference neighbors and computes overlap/concordance.

    Both DataFrames must have: id, smiles, and target columns.

    Use ``concordance_results()`` to get a unified DataFrame with x, y coordinates,
    dataset labels, and concordance columns (overlap, residual).
    """

    def __init__(
        self,
        df_reference: pd.DataFrame,
        df_query: pd.DataFrame,
        target_column: str,
        id_column: str,
        k_neighbors: int = 5,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        """Initialize the DatasetConcordance analysis.

        Args:
            df_reference (pd.DataFrame): Reference dataset (must have id, smiles, and target columns)
            df_query (pd.DataFrame): Query dataset (must have id, smiles, and target columns)
            target_column (str): Name of the target column to compare (must exist in both DataFrames)
            id_column (str): Name of the ID column
            k_neighbors (int): Number of cross-dataset neighbors for SAR concordance (default: 5)
            radius (int): Morgan fingerprint radius (default: 2 = ECFP4)
            n_bits (int): Number of fingerprint bits (default: 2048)
        """
        self.id_column = id_column
        self.target_column = target_column
        self.k_neighbors = k_neighbors

        # Validate required columns in both DataFrames
        required = {id_column, "smiles", target_column}
        for label, df in [("Reference", df_reference), ("Query", df_query)]:
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{label} DataFrame missing columns: {missing}")

        # Combine with a 'dataset' bookkeeping column
        df_reference = df_reference.copy()
        df_query = df_query.copy()

        # Deduplicate IDs within each dataset (keep first, log details)
        for label, df in [("Reference", df_reference), ("Query", df_query)]:
            dup_mask = df.duplicated(subset=id_column, keep="first")
            if dup_mask.any():
                dup_rows = df[df[id_column].isin(df.loc[dup_mask, id_column])]
                log.warning(f"{label}: Dropping {dup_mask.sum()} duplicate IDs (keeping first):")
                for dup_id, group in dup_rows.groupby(id_column):
                    targets = group[target_column].tolist()
                    log.warning(f"  {dup_id}: {target_column}={targets}")
                df.drop(df[dup_mask].index, inplace=True)

        df_reference["dataset"] = "reference"
        df_query["dataset"] = "query"
        df_combined = pd.concat([df_reference, df_query], ignore_index=True)

        log.info(f"Dataset: {len(df_combined)} compounds ({len(df_reference)} reference, {len(df_query)} query)")
        log.info(f"Target column: {target_column}")

        # Build ONE FingerprintProximity on the combined data
        # (shared fingerprints, NN model, UMAP projection)
        self._prox = FingerprintProximity(
            df_combined,
            id_column=id_column,
            target=target_column,
            include_all_columns=True,
            radius=radius,
            n_bits=n_bits,
        )

        # Compute cross-dataset concordance
        log.info("Computing cross-dataset concordance...")
        self._concordance_df = self._compute_concordance()

        n_total = len(self._concordance_df)
        log.info(f"Cross-dataset mean NN similarity: {self._concordance_df['tanimoto_sim'].mean():.3f}")
        log.info(f"Concordance computed for {n_total} query compounds")

    def concordance_results(self) -> pd.DataFrame:
        """Return a unified DataFrame with coordinates, dataset labels, and concordance info.

        Merges the per-query concordance columns (overlap, residual) into the
        combined proximity DataFrame. Reference compounds get NaN for concordance columns.

        Returns:
            pd.DataFrame: Unified DataFrame with these added columns:
                - dataset: "reference" or "query"
                - x, y: UMAP 2D coordinates
                - tanimoto_sim: best Tanimoto similarity to any reference compound
                - target_residual: query target minus median target of nearest reference neighbors
        """
        df = self._prox.df.copy()

        # Left join concordance columns (query-only) onto full df
        concordance_cols = [self.id_column, "tanimoto_sim", "target_residual"]
        df = df.merge(self._concordance_df[concordance_cols], on=self.id_column, how="left")

        # Drop internal proximity columns not needed in concordance results
        internal_cols = ["nn_distance", "nn_id", "nn_target", "nn_target_diff", "nn_similarity", "fingerprint"]
        df = df.drop(columns=[c for c in internal_cols if c in df.columns])

        return df

    def query_neighbors(
        self,
        query_id: str,
        n_neighbors: int = 5,
        include_self: bool = True,
    ) -> pd.DataFrame:
        """Get the nearest reference neighbors for a query compound.

        Useful for drilling into a specific compound's neighborhood — e.g., hover
        details in the UI showing the closest reference compounds with links.

        Args:
            query_id (str): ID of a query compound
            n_neighbors (int): Number of reference neighbors to return (default: 5)
            include_self (bool): Include the query compound in results (default: True)

        Returns:
            pd.DataFrame: Reference neighbors sorted by similarity (descending),
                with columns from prox.df (id, similarity, smiles, target, etc.)
        """
        # Get extra neighbors to account for same-dataset hits being filtered out
        k = n_neighbors * 3 + (1 if include_self else 0)
        neighbors_df = self._prox.neighbors([query_id], n_neighbors=k, include_self=include_self)

        # Filter to reference neighbors (keep self if included)
        dataset_lookup = self._prox.df.set_index(self.id_column)["dataset"]
        neighbors_df["neighbor_dataset"] = neighbors_df["neighbor_id"].map(dataset_lookup)
        is_self = neighbors_df["neighbor_id"] == query_id
        is_ref = neighbors_df["neighbor_dataset"] == "reference"
        result = neighbors_df[is_self | is_ref].drop(columns=["neighbor_dataset"])

        # Trim to requested count
        keep = n_neighbors + (1 if include_self else 0)
        return result.head(keep).reset_index(drop=True)

    def _compute_concordance(self) -> pd.DataFrame:
        """Compute overlap and SAR concordance for each query compound vs the reference.

        For each query compound:
            1. Get neighbors from the combined model
            2. Filter to reference-only neighbors
            3. Compute overlap (best Tanimoto) and concordance (median target residual)

        Returns:
            pd.DataFrame: Per-query-compound concordance with columns:
                id, tanimoto_sim, target_residual
        """
        # Get query compound IDs and targets from the combined prox.df
        query_mask = self._prox.df["dataset"] == "query"
        query_ids = self._prox.df.loc[query_mask, self.id_column].tolist()
        query_targets = self._prox.df.loc[query_mask].set_index(self.id_column)[self.target_column]

        # Neighbor lookup (vectorized — all query compounds in one call)
        n_neighbors = 50
        all_neighbors = self._prox.neighbors(query_ids, n_neighbors=n_neighbors)

        # Filter to reference-only neighbors
        dataset_lookup = self._prox.df.set_index(self.id_column)["dataset"]
        all_neighbors["neighbor_dataset"] = all_neighbors["neighbor_id"].map(dataset_lookup)
        ref_neighbors = all_neighbors[all_neighbors["neighbor_dataset"] == "reference"].copy()

        # Best Tanimoto similarity per query compound
        best_sim = ref_neighbors.groupby(self.id_column)["similarity"].max()

        # Target residual: median residual from top-k reference neighbors
        ref_neighbors["_rank"] = ref_neighbors.groupby(self.id_column)["similarity"].rank(
            method="first", ascending=False
        )
        top_k = ref_neighbors[ref_neighbors["_rank"] <= self.k_neighbors]
        neighbor_medians = top_k.groupby(self.id_column)[self.target_column].median()

        # Build result DataFrame
        results = pd.DataFrame({self.id_column: query_ids})
        results["tanimoto_sim"] = results[self.id_column].map(best_sim).fillna(0.0)

        # target_residual = query target - median reference target
        q_targets = results[self.id_column].map(query_targets)
        ref_medians = results[self.id_column].map(neighbor_medians)
        results["target_residual"] = q_targets - ref_medians

        return results


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    from workbench.utils.test_data_generator import TestDataGenerator

    # Pandas Options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)

    test_data = TestDataGenerator()

    # Get reference and query DataFrames
    ref_df, query_df = test_data.aqsol_alignment_data(overlap="medium", alignment="high")
    print(f"Reference: {len(ref_df)}, Query: {len(query_df)}")

    dc = DatasetConcordance(
        ref_df,
        query_df,
        target_column="solubility",
        id_column="id",
    )

    # Get the unified DataFrame
    df = dc.concordance_results()
    print(f"\nUnified DF shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dataset counts:\n{df['dataset'].value_counts()}")
    print("\nQuery compounds (first 10):")
    query_cols = ["id", "dataset", "x", "y", "tanimoto_sim", "target_residual"]
    print(df[df["dataset"] == "query"][query_cols].head(10))

    # Test query_neighbors — drill into a specific query compound
    query_id = df[df["dataset"] == "query"].iloc[0][dc.id_column]
    print(f"\nQuery neighbors for '{query_id}':")
    print(dc.query_neighbors(query_id, n_neighbors=5))

    print("\nDatasetConcordance tests completed!")
