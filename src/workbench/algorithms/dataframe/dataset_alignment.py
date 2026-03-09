"""Dataset Alignment: Compare two molecular datasets for overlap and target agreement.

Takes reference and query DataFrames, combines them with a ``dataset`` bookkeeping
column, and builds one FingerprintProximity model on the combined data. This gives
a shared UMAP projection and a single NN model that can be queried for cross-dataset
neighbors by filtering on the ``dataset`` column.

Produces:
    - **alignment_df**: Per-query-compound similarity and target residuals.

This DataFrame, plus the shared 2D coordinates in ``self._prox.df``, back the
scatter+contour visualization UI.

Use cases:
    - Data fusion: Can proprietary and public ADMET data be safely merged for training?
    - Assay alignment: Do two assays measuring the same endpoint agree?
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


class DatasetAlignment:
    """Compare two molecular datasets for chemical space overlap and target value alignment.

    Takes reference and query DataFrames, combines them internally, and builds a single
    FingerprintProximity model (shared fingerprints, NN model, and UMAP projection).
    For each query compound, finds reference neighbors and computes similarity/residuals.

    Both DataFrames must have: id, smiles, and target columns.

    Use ``dataset_alignment_results()`` to get a unified DataFrame with x, y coordinates,
    dataset labels, and alignment columns (similarity, residuals).
    """

    def __init__(
        self,
        df_reference: pd.DataFrame,
        df_query: pd.DataFrame,
        target_column: str,
        id_column: str,
        k_neighbors: int = 5,
        overlap_thres: float = 0.6,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        """Initialize the DatasetAlignment analysis.

        Args:
            df_reference (pd.DataFrame): Reference dataset (must have id, smiles, and target columns)
            df_query (pd.DataFrame): Query dataset (must have id, smiles, and target columns)
            target_column (str): Name of the target column to compare (must exist in both DataFrames)
            id_column (str): Name of the ID column
            k_neighbors (int): Number of cross-dataset neighbors for target alignment (default: 5)
            overlap_thres (float): Minimum Tanimoto similarity for computing median_ref_residual
                (default: 0.6). Query compounds below this get NaN residuals.
            radius (int): Morgan fingerprint radius (default: 2 = ECFP4)
            n_bits (int): Number of fingerprint bits (default: 2048)
        """
        self.id_column = id_column
        self.target_column = target_column
        self.k_neighbors = k_neighbors
        self.overlap_thres = overlap_thres

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

        # Compute cross-dataset target alignment
        log.info("Computing cross-dataset alignment...")
        self._alignment_df = self._compute_alignment()

        n_total = len(self._alignment_df)
        log.info(f"Cross-dataset mean NN similarity: {self._alignment_df['highest_ref_tanimoto'].mean():.3f}")
        log.info(f"Alignment computed for {n_total} query compounds")

    def dataset_alignment_results(self) -> pd.DataFrame:
        """Return a unified DataFrame with coordinates, dataset labels, and alignment info.

        Merges the per-query alignment columns (similarity, residuals) into the
        combined proximity DataFrame. Reference compounds get NaN for alignment columns.

        Returns:
            pd.DataFrame: Unified DataFrame with these added columns:
                - dataset: "reference" or "query"
                - x, y: UMAP 2D coordinates
                - highest_ref_tanimoto: best Tanimoto similarity to any reference compound
                - median_ref_residual: query target minus median target of nearest reference neighbors
        """
        df = self._prox.df.copy()

        # Left join alignment columns (query-only) onto full df
        alignment_cols = [self.id_column, "highest_ref_tanimoto", "median_ref_residual"]
        df = df.merge(self._alignment_df[alignment_cols], on=self.id_column, how="left")

        # Drop internal proximity columns not needed in alignment results
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

    def _compute_alignment(self) -> pd.DataFrame:
        """Compute target alignment for each query compound vs the reference.

        For each query compound:
            1. Get neighbors from the combined model
            2. Filter to reference-only neighbors
            3. Compute highest similarity and median target residual from top-k neighbors

        Returns:
            pd.DataFrame: Per-query-compound alignment with columns:
                id, highest_ref_tanimoto, median_ref_residual
        """
        # Get query compound IDs from the combined prox.df
        query_mask = self._prox.df["dataset"] == "query"
        query_ids = self._prox.df.loc[query_mask, self.id_column].tolist()
        query_targets = self._prox.df.loc[query_mask].set_index(self.id_column)[self.target_column]

        # Get neighbors for all query compounds (enough to filter down to reference)
        n_neighbors = max(20, self.k_neighbors * 4)
        all_neighbors = self._prox.neighbors(query_ids, n_neighbors=n_neighbors)

        # Filter to reference-only neighbors
        dataset_lookup = self._prox.df.set_index(self.id_column)["dataset"]
        all_neighbors["neighbor_dataset"] = all_neighbors["neighbor_id"].map(dataset_lookup)
        ref_neighbors = all_neighbors[all_neighbors["neighbor_dataset"] == "reference"]

        # Build per-query alignment
        results = []
        for q_id in query_ids:
            q_target = query_targets.get(q_id)
            q_ref = ref_neighbors[ref_neighbors[self.id_column] == q_id]

            # Best reference neighbor similarity
            best_sim = float(q_ref["similarity"].max()) if len(q_ref) > 0 else 0.0

            # Median target residual from top-k reference neighbors (only above overlap threshold)
            above_thres = q_ref[q_ref["similarity"] >= self.overlap_thres]
            if len(above_thres) > 0 and pd.notna(q_target):
                top_k = above_thres.nlargest(min(self.k_neighbors, len(above_thres)), "similarity")
                neighbor_median = float(top_k[self.target_column].dropna().median())
                residual = float(q_target) - neighbor_median
            else:
                residual = float("nan")

            results.append({
                self.id_column: q_id,
                "highest_ref_tanimoto": best_sim,
                "median_ref_residual": residual,
            })

        return pd.DataFrame(results)


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

    da = DatasetAlignment(
        ref_df,
        query_df,
        target_column="solubility",
        id_column="id",
    )

    # Get the unified DataFrame
    df = da.dataset_alignment_results()
    print(f"\nUnified DF shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dataset counts:\n{df['dataset'].value_counts()}")
    print(f"\nQuery compounds (first 10):")
    query_cols = ["id", "dataset", "x", "y", "highest_ref_tanimoto", "median_ref_residual"]
    print(df[df["dataset"] == "query"][query_cols].head(10))

    # Test query_neighbors — drill into a specific query compound
    query_id = df[df["dataset"] == "query"].iloc[0][da.id_column]
    print(f"\nQuery neighbors for '{query_id}':")
    print(da.query_neighbors(query_id, n_neighbors=5))

    print("\nDatasetAlignment tests completed!")
