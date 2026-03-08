"""Dataset Alignment: Compare two molecular datasets for overlap and target agreement.

Takes reference and query DataFrames, combines them with a ``dataset`` bookkeeping
column, and builds one FingerprintProximity model on the combined data. This gives
a shared UMAP projection and a single NN model that can be queried for cross-dataset
neighbors by filtering on the ``dataset`` column.

Produces:
    - **overlap_df**: Per-query-compound 1-NN similarity to nearest reference neighbor
    - **alignment_df**: Per-query-compound target residuals vs K reference neighbors

These DataFrames, plus the shared 2D coordinates in ``self.prox.df``, back the
scatter+contour visualization UI.

Use cases:
    - Data fusion: Can proprietary and public ADMET data be safely merged for training?
    - Assay alignment: Do two assays measuring the same endpoint agree?
    - Model monitoring: Has the target relationship drifted in new data?

Statistical summaries (KS, JSD, PSI, t-test) and earlier diagnostic plots are
preserved in dataset_alignment_plots.py and utils/distribution_stats.py.

References:
    - Landrum & Riniker (2024) "Combining IC50 or Ki Values from Different Sources
      Is a Source of Significant Noise" JCIM
    - Parrondo-Pizarro et al. (2025) "Enhancing molecular property prediction through
      data integration and consistency assessment" J. Cheminform.
"""

import logging

import numpy as np
import pandas as pd

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

# Set up logging
log = logging.getLogger("workbench")


class DatasetAlignment:
    """Compare two molecular datasets for chemical space overlap and target value alignment.

    Takes reference and query DataFrames, combines them internally, and builds a single
    FingerprintProximity model (shared fingerprints, NN model, and UMAP projection),
    then uses cross-dataset neighbor queries for overlap and alignment scoring.

    Both DataFrames must have: id, smiles, and target columns.

    Attributes:
        prox: FingerprintProximity instance built on the combined dataset.
            ``prox.df`` has x, y, dataset, target, nn_similarity for all compounds.
        overlap_df: Per-query-compound 1-NN similarity to reference (chemical space overlap)
        alignment_df: Per-compound target residuals vs K reference neighbors (target alignment)
    """

    DATASET_COL = "dataset"
    DATASET_REF = "reference"
    DATASET_QUERY = "query"
    REQUIRED_COLUMNS = {"id", "smiles"}

    def __init__(
        self,
        df_reference: pd.DataFrame,
        df_query: pd.DataFrame,
        target_column: str,
        id_column: str = "id",
        k_neighbors: int = 5,
        min_similarity: float = 0.3,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        """Initialize the DatasetAlignment analysis.

        Args:
            df_reference (pd.DataFrame): Reference dataset (must have id, smiles, and target columns)
            df_query (pd.DataFrame): Query dataset (must have id, smiles, and target columns)
            target_column (str): Name of the target column to compare (must exist in both DataFrames)
            id_column (str): Name of the ID column (default: "id")
            k_neighbors (int): Number of cross-dataset neighbors for target alignment (default: 5)
            min_similarity (float): Minimum Tanimoto similarity to include in target alignment
                analysis (default: 0.3). Compounds below this threshold are excluded from
                concept shift assessment since they lack comparable reference compounds.
            radius (int): Morgan fingerprint radius (default: 2 = ECFP4)
            n_bits (int): Number of fingerprint bits (default: 2048)
        """
        self.id_column = id_column
        self.target_column = target_column
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity

        # Validate required columns in both DataFrames
        required = self.REQUIRED_COLUMNS | {target_column}
        for label, df in [("Reference", df_reference), ("Query", df_query)]:
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{label} DataFrame missing columns: {missing}")

        # Combine with a 'dataset' bookkeeping column
        df_reference = df_reference.copy()
        df_query = df_query.copy()
        df_reference[self.DATASET_COL] = self.DATASET_REF
        df_query[self.DATASET_COL] = self.DATASET_QUERY
        df_combined = pd.concat([df_reference, df_query], ignore_index=True)

        log.info(f"Dataset: {len(df_combined)} compounds ({len(df_reference)} reference, {len(df_query)} query)")
        log.info(f"Target column: {target_column}")

        # Build ONE FingerprintProximity on the combined data
        # (shared fingerprints, NN model, UMAP projection)
        self.prox = FingerprintProximity(
            df_combined,
            id_column=id_column,
            target=target_column,
            include_all_columns=True,
            radius=radius,
            n_bits=n_bits,
        )

        # Compute cross-dataset overlap and target alignment
        log.info("Computing cross-dataset neighbors for overlap and alignment...")
        self.overlap_df, self.alignment_df = self._compute_alignment()

        n_comparable = len(self.alignment_df)
        n_excluded = len(df_query) - n_comparable
        log.info(f"Cross-dataset mean NN similarity: {self.overlap_df['tanimoto_similarity'].mean():.3f}")
        log.info(f"Target alignment: {n_comparable} comparable, {n_excluded} excluded (below min_similarity)")

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
        return self._cross_dataset_neighbors(
            [query_id], n_neighbors=n_neighbors, source_dataset=self.DATASET_QUERY,
            include_self=include_self,
        )

    def _cross_dataset_neighbors(
        self,
        query_ids: list,
        n_neighbors: int,
        source_dataset: str,
        include_self: bool = False,
    ) -> pd.DataFrame:
        """Get neighbors from the OTHER dataset.

        Queries the combined NN model with 3x the requested K to account for
        same-dataset hits being filtered out.

        Args:
            query_ids (list): IDs to find cross-dataset neighbors for
            n_neighbors (int): Number of cross-dataset neighbors desired per compound
            source_dataset (str): Dataset label of the query compounds ("reference" or "query")
            include_self (bool): Include query compounds in results (default: False)

        Returns:
            pd.DataFrame: Neighbor results filtered to only cross-dataset hits
                (plus self if include_self), trimmed to n_neighbors per query compound.
        """
        target_dataset = self.DATASET_REF if source_dataset == self.DATASET_QUERY else self.DATASET_QUERY

        # Build dataset lookup for filtering
        dataset_lookup = self.prox.df.set_index(self.id_column)[self.DATASET_COL]

        # Query with 3x K to have enough after filtering out same-dataset hits
        # +1 if include_self so we still return n_neighbors cross-dataset hits
        k = (n_neighbors + (1 if include_self else 0)) * 3
        neighbors_df = self.prox.neighbors(query_ids, n_neighbors=k, include_self=include_self)

        # Filter to cross-dataset neighbors (and self if included), trim to n_neighbors per query
        neighbors_df["neighbor_dataset"] = neighbors_df["neighbor_id"].map(dataset_lookup)
        query_id_set = set(query_ids)
        is_self = neighbors_df["neighbor_id"].isin(query_id_set)
        is_cross = neighbors_df["neighbor_dataset"] == target_dataset
        cross = neighbors_df[is_self | is_cross].copy()
        cross = cross.drop(columns=["neighbor_dataset"])
        # +1 to keep self row alongside n_neighbors cross-dataset hits
        keep = n_neighbors + (1 if include_self else 0)
        cross = cross.groupby(self.id_column).head(keep).reset_index(drop=True)
        return cross

    def _compute_alignment(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute overlap and target alignment using cross-dataset neighbors.

        Returns:
            tuple: (overlap_df, alignment_df)
                - overlap_df: Per-query 1-NN similarity to reference
                - alignment_df: Per-query target residuals vs K reference neighbors
        """
        # Get query compound IDs from the combined prox.df
        query_mask = self.prox.df[self.DATASET_COL] == self.DATASET_QUERY
        query_ids = self.prox.df.loc[query_mask, self.id_column].tolist()

        # Single cross-dataset neighbor call with K neighbors
        cross_neighbors = self._cross_dataset_neighbors(
            query_ids, n_neighbors=self.k_neighbors, source_dataset=self.DATASET_QUERY,
        )

        # --- Overlap DF: extract 1-NN per query ---
        # neighbors() returns sorted by similarity descending, so first per group is best
        best_neighbors = cross_neighbors.groupby(self.id_column).first().reset_index()
        overlap_df = best_neighbors[[self.id_column, "neighbor_id", "similarity"]].rename(
            columns={"neighbor_id": "nearest_neighbor_id", "similarity": "tanimoto_similarity"}
        )

        # --- Alignment DF: compute target residuals from K neighbors ---
        query_targets = self.prox.df.loc[query_mask].set_index(self.id_column)[self.target_column]
        alignment_results = []

        for q_id, group in cross_neighbors.groupby(self.id_column):
            q_target = query_targets.get(q_id)
            if pd.isna(q_target):
                continue

            # Nearest neighbor similarity (for min_similarity filtering)
            nn_similarity = group["similarity"].max()
            if nn_similarity < self.min_similarity:
                continue

            # Median target from K reference neighbors
            neighbor_targets = group[self.target_column].dropna()
            if len(neighbor_targets) == 0:
                continue

            neighbor_median = float(neighbor_targets.median())
            residual = float(q_target) - neighbor_median

            alignment_results.append({
                self.id_column: q_id,
                "query_target": float(q_target),
                "nearest_neighbor_id": group.loc[group["similarity"].idxmax(), "neighbor_id"],
                "tanimoto_similarity": nn_similarity,
                "neighbor_median_target": neighbor_median,
                "target_residual": residual,
            })

        alignment_df = pd.DataFrame(alignment_results)

        return overlap_df, alignment_df


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    from workbench.utils.test_data_generator import TestDataGenerator

    test_data = TestDataGenerator()

    # Get reference and query DataFrames
    ref_df, query_df = test_data.aqsol_alignment_data(overlap="high", alignment="high")
    print(f"Reference: {len(ref_df)}, Query: {len(query_df)}")

    da = DatasetAlignment(
        ref_df,
        query_df,
        target_column="solubility",
    )

    # Check the combined prox.df has both datasets with x, y coordinates
    print(f"\nProx DF shape: {da.prox.df.shape}")
    print(f"Columns: {list(da.prox.df.columns)}")
    print(f"Dataset counts:\n{da.prox.df['dataset'].value_counts()}")
    print(f"Has x,y: {'x' in da.prox.df.columns and 'y' in da.prox.df.columns}")

    print(f"\nOverlap DF shape: {da.overlap_df.shape}")
    print(f"Alignment DF shape: {da.alignment_df.shape}")
    print(f"\nOverlap DF head:\n{da.overlap_df.head()}")
    print(f"\nAlignment DF head:\n{da.alignment_df.head()}")

    # Test query_neighbors — drill into a specific query compound
    query_id = da.overlap_df.iloc[0][da.id_column]
    print(f"\nQuery neighbors for '{query_id}':")
    print(da.query_neighbors(query_id, n_neighbors=5))

    print("\nDatasetAlignment tests completed!")
