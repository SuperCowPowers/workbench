"""Dataset Alignment: Compare two molecular datasets for overlap and target agreement.

Builds a FingerprintProximity model on a reference dataset, then queries with
SMILES from a query dataset to compute:

    - **Chemical Space Overlap**: Per-query-compound nearest-neighbor Tanimoto
      similarity to the reference (overlap_df)
    - **Target Alignment**: Per-compound target residuals between query and
      K nearest reference neighbors (alignment_df)

These DataFrames back the scatter+contour visualization UI: reference contours
overlaid with query compounds colored by alignment quality.

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

    Builds a FingerprintProximity model on the reference dataset, then queries with
    SMILES from the query dataset to compute overlap and target alignment.

    Attributes:
        prox: FingerprintProximity instance built on the reference dataset
        overlap_df: Per-query-compound 1-NN similarity to reference (chemical space overlap)
        alignment_df: Per-compound target residuals vs K reference neighbors (target alignment)
    """

    def __init__(
        self,
        df_reference: pd.DataFrame,
        df_query: pd.DataFrame,
        target_column: str,
        id_column_reference: str = "id",
        id_column_query: str = "id",
        k_neighbors: int = 5,
        min_similarity: float = 0.3,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        """Initialize the DatasetAlignment analysis.

        Args:
            df_reference (pd.DataFrame): Reference dataset (must contain SMILES and target columns)
            df_query (pd.DataFrame): Query dataset (must contain SMILES and target columns)
            target_column (str): Name of the target column to compare (must exist in both DataFrames)
            id_column_reference (str): ID column name in df_reference
            id_column_query (str): ID column name in df_query
            k_neighbors (int): Number of neighbors for median target computation (default: 5)
            min_similarity (float): Minimum Tanimoto similarity to include in target alignment
                analysis (default: 0.3). Compounds below this threshold are excluded from
                concept shift assessment since they lack comparable reference compounds.
            radius (int): Morgan fingerprint radius (default: 2 = ECFP4)
            n_bits (int): Number of fingerprint bits (default: 2048)
        """
        self.target_column = target_column
        self.id_column_reference = id_column_reference
        self.id_column_query = id_column_query
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self._radius = radius
        self._n_bits = n_bits

        # Store copies of the dataframes
        self.df_reference = df_reference.copy()
        self.df_query = df_query.copy()

        # Validate SMILES columns
        self._smiles_col_reference = self._find_smiles_column(self.df_reference)
        self._smiles_col_query = self._find_smiles_column(self.df_query)
        if self._smiles_col_reference is None:
            raise ValueError("Reference dataset must have a SMILES column")
        if self._smiles_col_query is None:
            raise ValueError("Query dataset must have a SMILES column")

        # Validate target column
        if target_column not in self.df_reference.columns:
            raise ValueError(f"Target column '{target_column}' not found in reference dataset")
        if target_column not in self.df_query.columns:
            raise ValueError(f"Target column '{target_column}' not found in query dataset")

        log.info(f"Reference dataset: {len(self.df_reference)} compounds")
        log.info(f"Query dataset: {len(self.df_query)} compounds")
        log.info(f"Target column: {target_column}")

        # Build FingerprintProximity on reference dataset with target
        self.prox = FingerprintProximity(
            self.df_reference,
            id_column=id_column_reference,
            target=target_column,
            radius=radius,
            n_bits=n_bits,
        )

        # Single neighbors_from_smiles call for both overlap and target alignment
        query_smiles = self.df_query[self._smiles_col_query].tolist()
        log.info(f"Computing nearest neighbors in reference for {len(query_smiles)} query compounds")
        self._all_neighbors_df = self.prox.neighbors_from_smiles(
            query_smiles, n_neighbors=max(1, self.k_neighbors)
        )

        # Compute cross-dataset overlap (1-NN for chemical space analysis)
        self.overlap_df = self._compute_cross_dataset_overlap()

        log.info(f"Reference within-dataset mean NN similarity: {self.prox.df['nn_similarity'].mean():.3f}")
        log.info(f"Cross-dataset mean NN similarity: {self.overlap_df['tanimoto_similarity'].mean():.3f}")

        # Compute target alignment (K-NN for concept shift analysis)
        self.alignment_df = self._compute_target_alignment()

        n_comparable = len(self.alignment_df)
        n_excluded = len(self.df_query) - n_comparable
        log.info(f"Target alignment: {n_comparable} comparable compounds, {n_excluded} excluded (below min_similarity)")

    @staticmethod
    def _find_smiles_column(df: pd.DataFrame) -> str | None:
        """Find the SMILES column in a DataFrame (case-insensitive).

        Args:
            df (pd.DataFrame): DataFrame to search

        Returns:
            str | None: Column name if found, None otherwise
        """
        for col in df.columns:
            if col.lower() == "smiles":
                return col
        return None

    def _compute_cross_dataset_overlap(self) -> pd.DataFrame:
        """For each query compound, find nearest neighbor in reference.

        Uses the pre-computed K-NN results (self._all_neighbors_df), extracting
        just the top-1 neighbor per query for chemical space overlap analysis.

        Returns:
            pd.DataFrame: DataFrame with columns: id, smiles, nearest_neighbor_id,
                tanimoto_similarity, nearest_neighbor_smiles
        """
        query_smiles = self.df_query[self._smiles_col_query].tolist()
        query_ids = self.df_query[self.id_column_query].tolist()

        # Extract 1-NN from the pre-computed K-NN results (first row per query = highest similarity)
        neighbors_df = self._all_neighbors_df

        results = []
        for q_id, q_smi in zip(query_ids, query_smiles):
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
                results.append(
                    {
                        "id": q_id,
                        "smiles": q_smi,
                        "nearest_neighbor_id": None,
                        "tanimoto_similarity": 0.0,
                    }
                )

        result_df = pd.DataFrame(results)

        # Add nearest neighbor SMILES from reference (drop_duplicates handles repeated IDs)
        ref_smiles_map = self.df_reference.drop_duplicates(subset=self.id_column_reference).set_index(
            self.id_column_reference
        )[self._smiles_col_reference]
        result_df["nearest_neighbor_smiles"] = result_df["nearest_neighbor_id"].map(ref_smiles_map)

        return result_df.sort_values("tanimoto_similarity", ascending=False).reset_index(drop=True)

    def _compute_target_alignment(self) -> pd.DataFrame:
        """Compute per-compound target alignment using K nearest neighbors.

        For each query compound, finds K nearest neighbors in the reference dataset,
        computes the median target value of those neighbors, and compares it to the
        query compound's target. This is a cross-dataset extension of HTG analysis.

        Only includes compounds where the nearest neighbor Tanimoto similarity meets
        the min_similarity threshold — ensuring we only assess concept shift where
        the datasets actually overlap in chemical space.

        Returns:
            pd.DataFrame: Per-compound alignment with columns: id, smiles, query_target,
                nearest_neighbor_id, tanimoto_similarity, neighbor_median_target, target_residual
        """
        log.info(f"Computing target alignment (k={self.k_neighbors}, min_sim={self.min_similarity})")

        query_smiles = self.df_query[self._smiles_col_query].tolist()
        query_ids = self.df_query[self.id_column_query].tolist()
        query_targets = self.df_query[self.target_column].tolist()

        # Use pre-computed K-NN results
        neighbors_df = self._all_neighbors_df

        results = []
        for q_id, q_smi, q_target in zip(query_ids, query_smiles, query_targets):
            # Skip if query target is NaN
            if pd.isna(q_target):
                continue

            # Get all K neighbors for this query compound
            match = neighbors_df[neighbors_df["query_id"] == q_smi]
            if len(match) == 0:
                continue

            # Nearest neighbor similarity (for filtering)
            nn_similarity = match["similarity"].max()

            # Skip if below minimum similarity threshold
            if nn_similarity < self.min_similarity:
                continue

            # Compute median target from K neighbors (using target column returned by FP proximity)
            neighbor_targets = match[self.target_column].dropna()
            if len(neighbor_targets) == 0:
                continue

            neighbor_median_target = float(neighbor_targets.median())
            target_residual = float(q_target) - neighbor_median_target

            # Get the nearest neighbor ID
            nn_row = match.loc[match["similarity"].idxmax()]

            results.append(
                {
                    "id": q_id,
                    "smiles": q_smi,
                    "query_target": float(q_target),
                    "nearest_neighbor_id": nn_row["neighbor_id"],
                    "tanimoto_similarity": nn_similarity,
                    "neighbor_median_target": neighbor_median_target,
                    "target_residual": target_residual,
                }
            )

        return pd.DataFrame(results)


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    from workbench.utils.test_data_generator import TestDataGenerator

    test_data = TestDataGenerator()

    # Quick single-combo smoke test
    ref_df, query_df = test_data.aqsol_alignment_data(overlap="high", alignment="high")
    print(f"Reference: {len(ref_df)}, Query: {len(query_df)}")

    da = DatasetAlignment(
        ref_df,
        query_df,
        target_column="solubility",
        id_column_reference="id",
        id_column_query="id",
    )

    print(f"\nOverlap DF shape: {da.overlap_df.shape}")
    print(f"Alignment DF shape: {da.alignment_df.shape}")
    print(f"\nOverlap DF head:\n{da.overlap_df.head()}")
    print(f"\nAlignment DF head:\n{da.alignment_df.head()}")

    print("\nDatasetAlignment smoke test completed!")
