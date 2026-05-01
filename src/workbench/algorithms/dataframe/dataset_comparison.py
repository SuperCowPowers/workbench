"""Dataset Comparison: chemical-space overlap and SAR concordance between two molecular DataFrames.

Combines reference and query DataFrames with a ``dataset`` bookkeeping column and
builds a single FingerprintProximity model on the combined data — shared
fingerprints, shared KNN, shared UMAP. For each query compound it then finds
reference neighbors and computes:

    - **tanimoto_sim**: best Tanimoto similarity to any reference compound
    - **target_residual**: query target minus the median target of the top-k
      reference neighbors (the SAR concordance signal)

The reference and query targets do **not** have to share a column name —
e.g. comparing LogP (reference) against LogD (query) is supported via
``reference_target="logp"`` and ``query_target="logd"``. The residual is
always ``query_target_value - median(reference_target_values_of_neighbors)``.

Use cases:
    - Data fusion: Can proprietary and public ADMET data be safely merged?
    - Cross-endpoint analysis: Does an auxiliary task cover the primary task's
      chemical space well enough for multi-task learning to help?
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

log = logging.getLogger("workbench")


class DatasetComparison:
    """Compare two molecular datasets for chemical-space overlap and SAR concordance.

    Both DataFrames must have an id column, a ``smiles`` column, and their
    respective target columns. Use ``results()`` for the unified per-compound
    DataFrame, ``summary()`` for aggregate roll-up stats, and
    ``exact_smiles_overlap()`` for the canonical-SMILES inner join.

    For a quick label-only pre-flight on a pre-combined multi-task DataFrame
    (no fingerprint cost), see ``workbench.utils.multi_task.assess_multi_task_data``.
    """

    def __init__(
        self,
        df_reference: pd.DataFrame,
        df_query: pd.DataFrame,
        reference_target: str,
        query_target: str,
        id_column: str,
        k_neighbors: int = 5,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        """Initialize the comparison and run the cross-dataset concordance computation.

        Args:
            df_reference: Reference dataset — must have id_column, ``smiles``, and ``reference_target``.
            df_query: Query dataset — must have id_column, ``smiles``, and ``query_target``.
            reference_target: Target column name in the reference DataFrame.
            query_target: Target column name in the query DataFrame. May be the
                same name as ``reference_target`` (e.g. comparing two LogP datasets)
                or different (e.g. ``reference_target="logp"`` vs ``query_target="logd"``).
            id_column: Name of the ID column (must exist in both DataFrames).
            k_neighbors: Number of cross-dataset neighbors used for the residual
                computation (default: 5).
            radius: Morgan fingerprint radius (default: 2 = ECFP4).
            n_bits: Number of fingerprint bits (default: 2048).
        """
        self.id_column = id_column
        self.reference_target = reference_target
        self.query_target = query_target
        self.k_neighbors = k_neighbors

        # Validate required columns
        for label, df, target in [
            ("Reference", df_reference, reference_target),
            ("Query", df_query, query_target),
        ]:
            missing = {id_column, "smiles", target} - set(df.columns)
            if missing:
                raise ValueError(f"{label} DataFrame missing columns: {missing}")

        df_reference = df_reference.copy()
        df_query = df_query.copy()

        # Deduplicate IDs within each dataset (keep first, log details)
        for label, df, target in [
            ("Reference", df_reference, reference_target),
            ("Query", df_query, query_target),
        ]:
            dup_mask = df.duplicated(subset=id_column, keep="first")
            if dup_mask.any():
                dup_rows = df[df[id_column].isin(df.loc[dup_mask, id_column])]
                log.warning(f"{label}: Dropping {dup_mask.sum()} duplicate IDs (keeping first):")
                for dup_id, group in dup_rows.groupby(id_column):
                    targets = group[target].tolist()
                    log.warning(f"  {dup_id}: {target}={targets}")
                df.drop(df[dup_mask].index, inplace=True)

        # IDs must be disjoint across datasets — the combined model indexes by id_column
        shared_ids = set(df_reference[id_column]) & set(df_query[id_column])
        if shared_ids:
            example = list(shared_ids)[:5]
            raise ValueError(
                f"Reference and query share {len(shared_ids)} {id_column!r} value(s) "
                f"(examples: {example}). Prefix or namespace the IDs so they are disjoint."
            )

        df_reference["dataset"] = "reference"
        df_query["dataset"] = "query"
        df_combined = pd.concat([df_reference, df_query], ignore_index=True)

        log.info(f"Dataset: {len(df_combined)} compounds ({len(df_reference)} reference, {len(df_query)} query)")
        log.info(f"Reference target: {reference_target}    Query target: {query_target}")

        # Build ONE FingerprintProximity on the combined data
        # (shared fingerprints, KNN model, UMAP projection).
        # We pass target=None because the two datasets may have different target
        # columns; both columns are kept by include_all_columns=True.
        self._prox = FingerprintProximity(
            df_combined,
            id_column=id_column,
            target=None,
            include_all_columns=True,
            radius=radius,
            n_bits=n_bits,
        )

        log.info("Computing cross-dataset concordance...")
        self._concordance_df = self._compute_concordance()

        n_total = len(self._concordance_df)
        log.info(f"Cross-dataset mean NN similarity: {self._concordance_df['tanimoto_sim'].mean():.3f}")
        log.info(f"Concordance computed for {n_total} query compounds")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def results(self) -> pd.DataFrame:
        """Return a unified per-compound DataFrame with coordinates and concordance columns.

        Returns:
            pd.DataFrame with all reference and query rows, plus:
                - ``dataset``: "reference" or "query"
                - ``x``, ``y``: shared UMAP 2D coordinates
                - ``tanimoto_sim``: best Tanimoto similarity to any reference compound (NaN for ref rows)
                - ``target_residual``: query_target minus median reference target across the top-k
                  similar reference neighbors (NaN for ref rows)
            The reference and query target columns are both present; rows from
            the other dataset have NaN for the column that doesn't apply
            (unless ``reference_target == query_target``, in which case there's
            a single shared column).
        """
        df = self._prox.df.copy()

        concordance_cols = [self.id_column, "tanimoto_sim", "target_residual"]
        df = df.merge(self._concordance_df[concordance_cols], on=self.id_column, how="left")

        internal_cols = ["nn_distance", "nn_id", "nn_target", "nn_target_diff", "nn_similarity", "fingerprint"]
        df = df.drop(columns=[c for c in internal_cols if c in df.columns])

        return df

    def neighbors(self, compound_id: str, n_neighbors: int = 10, reference_only: bool = False) -> pd.DataFrame:
        """Get the nearest neighbors for a compound in the combined fingerprint space.

        Args:
            compound_id: ID of any compound (reference or query).
            n_neighbors: Number of neighbors to return (default: 10).
            reference_only: If True, only return reference-dataset neighbors (default: False).

        Returns:
            pd.DataFrame: Neighbors sorted by similarity (descending).
        """
        # Fetch extra neighbors when filtering to account for same-dataset hits
        k = n_neighbors * 3 if reference_only else n_neighbors
        nbrs = self._prox.neighbors(compound_id, n_neighbors=k)

        if reference_only:
            dataset_lookup = self._prox.df.set_index(self.id_column)["dataset"]
            nbrs["neighbor_dataset"] = nbrs["neighbor_id"].map(dataset_lookup)
            is_self = nbrs["neighbor_id"] == compound_id
            is_ref = nbrs["neighbor_dataset"] == "reference"
            nbrs = nbrs[is_self | is_ref].drop(columns=["neighbor_dataset"]).head(n_neighbors)

        cols = [self.id_column, "neighbor_id", "dataset", "similarity"]
        for t in {self.reference_target, self.query_target}:
            if t in nbrs.columns and t not in cols:
                cols.append(t)
        cols = [c for c in cols if c in nbrs.columns]
        return nbrs[cols].reset_index(drop=True)

    def exact_smiles_overlap(self) -> pd.DataFrame:
        """Inner-join reference and query on canonical SMILES.

        No fingerprint computation involved — pure SMILES intersection. Useful
        for the subset where both datasets report a value for literally the same
        compound.

        Returns:
            pd.DataFrame: One row per shared canonical SMILES, with:
                - ``smiles``
                - ``<id_column>_reference``, ``<id_column>_query``
                - ``<reference_target>``, ``<query_target>`` (both side-by-side)
            If reference_target == query_target the column is suffixed
            ``_reference`` / ``_query`` to keep both values.
        """
        ref_cols = [self.id_column, "smiles", self.reference_target]
        qry_cols = [self.id_column, "smiles", self.query_target]

        ref = self._prox.df.loc[self._prox.df["dataset"] == "reference", ref_cols].copy()
        qry = self._prox.df.loc[self._prox.df["dataset"] == "query", qry_cols].copy()

        suffixes = ("_reference", "_query")
        return ref.merge(qry, on="smiles", suffixes=suffixes)

    def summary(self, sim_thresholds: tuple = (0.9, 0.7, 0.5, 0.3)) -> dict:
        """Aggregate roll-up statistics over the comparison.

        Args:
            sim_thresholds: Descending Tanimoto thresholds for coverage buckets
                (default: (0.9, 0.7, 0.5, 0.3)).

        Returns:
            dict with these keys:
                - ``n_reference``, ``n_query``: row counts after dedup
                - ``tanimoto``: dict with ``mean``, ``median``, ``min``, ``max`` of
                  best query→reference similarity
                - ``coverage``: dict mapping ``">= 0.70"`` etc. to ``{"count": n, "fraction": f}``
                - ``residual``: dict with ``mean``, ``median``, ``abs_mean``, ``abs_p95``
                  over query compounds that have any reference neighbor
                - ``residual_by_sim_band``: list of dicts, one per
                  half-open band ``[lo, hi)``, with ``n``, ``residual_mean``,
                  ``residual_abs_mean``, ``residual_abs_p95``
                - ``exact_smiles_overlap``: dict with ``count``, ``fraction_of_query``,
                  ``fraction_of_reference``, and (when both sides have target values)
                  ``pearson``, ``mean_diff``, ``abs_mean_diff``
        """
        df = self._prox.df
        n_ref = int((df["dataset"] == "reference").sum())
        n_qry = int((df["dataset"] == "query").sum())

        sims = self._concordance_df["tanimoto_sim"].fillna(0.0)
        residuals = self._concordance_df["target_residual"].dropna()

        out: dict = {
            "n_reference": n_ref,
            "n_query": n_qry,
            "tanimoto": {
                "mean": float(sims.mean()),
                "median": float(sims.median()),
                "min": float(sims.min()),
                "max": float(sims.max()),
            },
            "coverage": {},
            "residual": {},
            "residual_by_sim_band": [],
            "exact_smiles_overlap": {},
        }

        # Coverage buckets — descending thresholds
        for t in sorted(sim_thresholds, reverse=True):
            n = int((sims >= t).sum())
            out["coverage"][f">= {t:.2f}"] = {
                "count": n,
                "fraction": float(n / len(sims)) if len(sims) else 0.0,
            }
        # Residual stats
        if len(residuals):
            out["residual"] = {
                "n": int(len(residuals)),
                "mean": float(residuals.mean()),
                "median": float(residuals.median()),
                "abs_mean": float(residuals.abs().mean()),
                "abs_p95": float(residuals.abs().quantile(0.95)),
            }

        # Residual by similarity band — uses thresholds as band edges
        edges = sorted(set([0.0, *sim_thresholds, 1.0001]))
        bands = list(zip(edges[:-1], edges[1:]))
        merged = self._concordance_df.dropna(subset=["target_residual"])
        for lo, hi in bands:
            mask = (merged["tanimoto_sim"] >= lo) & (merged["tanimoto_sim"] < hi)
            sub = merged.loc[mask, "target_residual"]
            if len(sub) == 0:
                continue
            out["residual_by_sim_band"].append(
                {
                    "sim_lo": float(lo),
                    "sim_hi": float(min(hi, 1.0)),
                    "n": int(len(sub)),
                    "residual_mean": float(sub.mean()),
                    "residual_abs_mean": float(sub.abs().mean()),
                    "residual_abs_p95": float(sub.abs().quantile(0.95)),
                }
            )

        # Exact-SMILES overlap stats
        overlap = self.exact_smiles_overlap()
        out["exact_smiles_overlap"]["count"] = int(len(overlap))
        out["exact_smiles_overlap"]["fraction_of_query"] = float(len(overlap) / n_qry) if n_qry else 0.0
        out["exact_smiles_overlap"]["fraction_of_reference"] = float(len(overlap) / n_ref) if n_ref else 0.0

        ref_col, qry_col = self._overlap_target_columns()
        if ref_col in overlap.columns and qry_col in overlap.columns and len(overlap) > 1:
            corr = float(overlap[[ref_col, qry_col]].corr().iloc[0, 1])
            diff = overlap[ref_col] - overlap[qry_col]
            out["exact_smiles_overlap"]["pearson"] = corr
            out["exact_smiles_overlap"]["mean_diff"] = float(diff.mean())
            out["exact_smiles_overlap"]["abs_mean_diff"] = float(diff.abs().mean())

        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _overlap_target_columns(self) -> tuple:
        """Column names for ref/query targets in the exact_smiles_overlap() output."""
        if self.reference_target == self.query_target:
            return (f"{self.reference_target}_reference", f"{self.query_target}_query")
        return (self.reference_target, self.query_target)

    def _compute_concordance(self) -> pd.DataFrame:
        """For each query compound: best Tanimoto sim to ref + median ref-neighbor target residual."""
        query_mask = self._prox.df["dataset"] == "query"
        query_ids = self._prox.df.loc[query_mask, self.id_column].tolist()
        query_targets = self._prox.df.loc[query_mask].set_index(self.id_column)[self.query_target]

        # Vectorized neighbor lookup for all query compounds
        n_neighbors = 50
        all_neighbors = self._prox.neighbors(query_ids, n_neighbors=n_neighbors)

        dataset_lookup = self._prox.df.set_index(self.id_column)["dataset"]
        all_neighbors["neighbor_dataset"] = all_neighbors["neighbor_id"].map(dataset_lookup)
        ref_neighbors = all_neighbors[all_neighbors["neighbor_dataset"] == "reference"].copy()

        best_sim = ref_neighbors.groupby(self.id_column)["similarity"].max()

        ref_neighbors["_rank"] = ref_neighbors.groupby(self.id_column)["similarity"].rank(
            method="first", ascending=False
        )
        top_k = ref_neighbors[ref_neighbors["_rank"] <= self.k_neighbors]
        # Reference-target column is populated for ref-dataset rows; NaN for query rows
        # (unless reference_target == query_target, in which case it's a single shared column).
        neighbor_medians = top_k.groupby(self.id_column)[self.reference_target].median()

        results = pd.DataFrame({self.id_column: query_ids})
        results["tanimoto_sim"] = results[self.id_column].map(best_sim).fillna(0.0)
        q_targets = results[self.id_column].map(query_targets)
        ref_medians = results[self.id_column].map(neighbor_medians)
        results["target_residual"] = q_targets - ref_medians
        return results


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    from workbench.utils.synthetic_data_generator import SyntheticDataGenerator

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)

    test_data = SyntheticDataGenerator()

    ref_df, query_df = test_data.aqsol_alignment_data(overlap="medium", alignment="high")
    print(f"Reference: {len(ref_df)}, Query: {len(query_df)}")

    dc = DatasetComparison(
        ref_df,
        query_df,
        reference_target="solubility",
        query_target="solubility",
        id_column="id",
    )

    df = dc.results()
    print(f"\nUnified DF shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dataset counts:\n{df['dataset'].value_counts()}")
    print("\nQuery compounds (first 10):")
    query_cols = ["id", "dataset", "x", "y", "tanimoto_sim", "target_residual"]
    print(df[df["dataset"] == "query"][query_cols].head(10))

    print("\nSummary:")
    import json

    print(json.dumps(dc.summary(), indent=2))

    print("\nDatasetComparison tests completed!")
