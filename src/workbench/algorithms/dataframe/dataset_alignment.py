"""Dataset Alignment Analysis: Covariate Shift and Concept Shift Detection

This module compares two molecular datasets to assess how well "aligned" they are,
supporting all model types (XGBoost, PyTorch, ChemProp) since every dataset has SMILES.

Two levels of analysis:

    1. **Covariate Shift** (Chemical Space): Do the datasets cover the same chemical space?
       Uses Tanimoto fingerprint similarity distributions with KS test, JSD, and PSI.

    2. **Concept Shift** (Target Alignment): When compounds are structurally similar across
       datasets, do their target values agree? This detects inter-assay variability,
       systematic offsets, or experimental differences. Uses cross-dataset nearest-neighbor
       target residual analysis with t-test and Wilcoxon.

The concept shift approach is essentially a cross-dataset extension of Workbench's
High Target Gradient (HTG) analysis: instead of finding neighbors with different targets
*within* a dataset, we find neighbors *across* datasets and compare their target values.

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

import numpy as np
import pandas as pd

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

# Set up logging
log = logging.getLogger("workbench")


class DatasetAlignment:
    """Compare two molecular datasets for chemical space overlap and target value alignment.

    Builds a FingerprintProximity model on the reference dataset, then queries with SMILES
    from the query dataset to find nearest neighbors. Performs two levels of analysis:

    1. **Covariate Shift**: Are the chemical spaces similar? (KS test, JSD, PSI on
       Tanimoto similarity distributions)
    2. **Concept Shift**: Do target values agree where compounds are structurally similar?
       (target residual analysis with statistical tests)

    Attributes:
        prox: FingerprintProximity instance built on the reference dataset
        overlap_df: Results DataFrame with per-compound similarity scores
        alignment_df: Per-compound target alignment results (concept shift analysis)
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

        # Compute cross-dataset overlap (1-NN for chemical space analysis)
        self.overlap_df = self._compute_cross_dataset_overlap()

        # Extract similarity distributions for covariate shift metrics
        self._ref_nn_similarities = self.prox.df["nn_similarity"].values
        self._cross_nn_similarities = self.overlap_df["tanimoto_similarity"].values

        log.info(f"Reference within-dataset mean NN similarity: {self._ref_nn_similarities.mean():.3f}")
        log.info(f"Cross-dataset mean NN similarity: {self._cross_nn_similarities.mean():.3f}")

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

        Returns:
            pd.DataFrame: DataFrame with columns: id, smiles, nearest_neighbor_id,
                tanimoto_similarity, nearest_neighbor_smiles
        """
        log.info(f"Computing nearest neighbors in reference for {len(self.df_query)} query compounds")

        query_smiles = self.df_query[self._smiles_col_query].tolist()
        query_ids = self.df_query[self.id_column_query].tolist()

        # Get 1-NN for chemical space overlap analysis
        neighbors_df = self.prox.neighbors_from_smiles(query_smiles, n_neighbors=1)

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

        # Get K neighbors for target comparison
        neighbors_df = self.prox.neighbors_from_smiles(query_smiles, n_neighbors=self.k_neighbors)

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

    # ---- Chemical Space Overlap ----

    def summary_stats(self) -> pd.DataFrame:
        """Return distribution statistics for nearest-neighbor Tanimoto similarities.

        Returns:
            pd.DataFrame: Descriptive statistics including percentiles
        """
        return (
            self.overlap_df["tanimoto_similarity"]
            .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            .to_frame()
        )

    def novel_compounds(self, threshold: float = 0.4) -> pd.DataFrame:
        """Return query compounds that are novel (low similarity to reference).

        Args:
            threshold (float): Maximum Tanimoto similarity to consider "novel" (default: 0.4)

        Returns:
            pd.DataFrame: Query compounds with similarity below threshold
        """
        novel = self.overlap_df[self.overlap_df["tanimoto_similarity"] < threshold].copy()
        return novel.sort_values("tanimoto_similarity", ascending=True).reset_index(drop=True)

    def similar_compounds(self, threshold: float = 0.7) -> pd.DataFrame:
        """Return query compounds that are similar to reference (high overlap).

        Args:
            threshold (float): Minimum Tanimoto similarity to consider "similar" (default: 0.7)

        Returns:
            pd.DataFrame: Query compounds with similarity above threshold
        """
        similar = self.overlap_df[self.overlap_df["tanimoto_similarity"] >= threshold].copy()
        return similar.sort_values("tanimoto_similarity", ascending=False).reset_index(drop=True)

    def overlap_fraction(self, threshold: float = 0.7) -> float:
        """Return fraction of query compounds that overlap with reference above similarity threshold.

        Args:
            threshold (float): Minimum Tanimoto similarity to consider "overlapping"

        Returns:
            float: Fraction of query compounds with nearest neighbor similarity >= threshold
        """
        n_overlapping = (self.overlap_df["tanimoto_similarity"] >= threshold).sum()
        return n_overlapping / len(self.overlap_df)

    # ---- Covariate Shift (Chemical Space Distribution) ----

    def _compute_binned_distributions(self, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Compute normalized binned distributions for both similarity arrays.

        Histograms both the within-reference and cross-dataset NN similarity distributions
        into the same bins over [0, 1], adds epsilon to avoid zero bins, and normalizes.
        Default of 10 bins (deciles) follows standard PSI practice and avoids sparse-bin
        artifacts with typical dataset sizes.

        Args:
            n_bins (int): Number of histogram bins over [0, 1]

        Returns:
            tuple[np.ndarray, np.ndarray]: (p, q) where p is the reference distribution and
                q is the cross-dataset distribution, both normalized to sum to 1
        """
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        epsilon = 1e-10

        p_counts, _ = np.histogram(self._ref_nn_similarities, bins=bins)
        q_counts, _ = np.histogram(self._cross_nn_similarities, bins=bins)

        p = (p_counts + epsilon) / (p_counts + epsilon).sum()
        q = (q_counts + epsilon) / (q_counts + epsilon).sum()

        return p, q

    def ks_test(self) -> dict:
        """Kolmogorov-Smirnov test comparing within-reference vs cross-dataset NN similarity distributions.

        The KS test measures the maximum distance between two empirical CDFs.
        A large test statistic (or small p-value) indicates the query dataset
        is drawn from a different distribution than the reference.

        Returns:
            dict: Keys are 'statistic' (float, 0-1), 'p_value' (float),
                and 'shift_detected' (bool, True if p_value < 0.05)
        """
        from scipy.stats import ks_2samp

        stat, p_value = ks_2samp(self._ref_nn_similarities, self._cross_nn_similarities)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "shift_detected": p_value < 0.05,
        }

    def jensen_shannon_divergence(self) -> float:
        """Jensen-Shannon Divergence between within-reference and cross-dataset NN similarity distributions.

        JSD is a symmetric, bounded (0 to 1) divergence measure. Values near 0 indicate
        the datasets occupy the same chemical space; values near 1 indicate completely
        different chemical spaces.

        Returns:
            float: JSD value in [0, 1]
        """
        from scipy.spatial.distance import jensenshannon

        p, q = self._compute_binned_distributions()
        # scipy returns sqrt(JSD) (the JS distance), so square it for true divergence
        # base=2 ensures JSD is bounded [0, 1]
        return float(jensenshannon(p, q, base=2) ** 2)

    def population_stability_index(self) -> float:
        """Population Stability Index comparing within-reference vs cross-dataset NN similarity distributions.

        PSI quantifies how much a distribution has shifted from a baseline. Commonly
        used in model monitoring to detect covariate shift.

        Interpretation:
            PSI < 0.1:  No significant shift
            0.1 <= PSI < 0.25:  Moderate shift, investigation recommended
            PSI >= 0.25:  Significant shift, action required

        Returns:
            float: PSI value (non-negative)
        """
        p, q = self._compute_binned_distributions()
        return float(np.sum((q - p) * np.log(q / p)))

    def covariate_shift_summary(self) -> dict:
        """Compute all covariate shift (chemical space distribution) metrics.

        Compares the within-reference nearest-neighbor similarity distribution against
        the cross-dataset nearest-neighbor similarity distribution using KS test,
        Jensen-Shannon Divergence, and Population Stability Index.

        Returns:
            dict: Dictionary with divergence metrics, severity labels, and distribution info
        """
        if len(self._ref_nn_similarities) < 20 or len(self._cross_nn_similarities) < 20:
            log.warning("Small sample size (< 20 compounds) may produce unreliable divergence metrics")

        ks = self.ks_test()
        jsd = self.jensen_shannon_divergence()
        psi = self.population_stability_index()

        if psi < 0.1:
            psi_severity = "none"
        elif psi < 0.25:
            psi_severity = "moderate"
        else:
            psi_severity = "significant"

        return {
            "ks_statistic": ks["statistic"],
            "ks_p_value": ks["p_value"],
            "jensen_shannon_divergence": jsd,
            "population_stability_index": psi,
            "shift_detected": ks["shift_detected"],
            "psi_severity": psi_severity,
            "ref_distribution_size": len(self._ref_nn_similarities),
            "query_distribution_size": len(self._cross_nn_similarities),
            "ref_mean_similarity": float(self._ref_nn_similarities.mean()),
            "query_mean_similarity": float(self._cross_nn_similarities.mean()),
        }

    # ---- Concept Shift (Target Alignment) ----

    def concept_shift_summary(self) -> dict:
        """Compute concept shift metrics: are target values aligned where datasets overlap?

        For each query compound with sufficient structural similarity to the reference,
        compares its target value to the median target of its K nearest neighbors in the
        reference. Statistical tests assess whether residuals are centered at zero (aligned)
        or systematically shifted (concept shift / assay offset).

        This is a cross-dataset extension of High Target Gradient (HTG) analysis.

        Returns:
            dict: Dictionary with keys:
                - mean_residual (float): Mean of target residuals (systematic offset direction)
                - median_residual (float): Median residual (robust to outliers)
                - std_residual (float): Standard deviation of residuals
                - mae (float): Mean absolute error (overall disagreement magnitude)
                - rmse (float): Root mean squared error
                - t_statistic (float): One-sample t-test statistic (H₀: mean=0)
                - t_p_value (float): t-test p-value
                - wilcoxon_statistic (float): Wilcoxon signed-rank test statistic
                - wilcoxon_p_value (float): Wilcoxon p-value
                - concept_shift_detected (bool): True if both tests reject H₀ at p < 0.05
                - n_comparable_compounds (int): Compounds with sufficient overlap for comparison
                - n_excluded_compounds (int): Compounds excluded (below min_similarity)
        """
        from scipy.stats import ttest_1samp, wilcoxon

        residuals = self.alignment_df["target_residual"].values
        n_comparable = len(residuals)
        n_excluded = len(self.df_query) - n_comparable

        if n_comparable < 5:
            log.warning(f"Only {n_comparable} comparable compounds — concept shift metrics unreliable")
            return {
                "mean_residual": float(np.mean(residuals)) if n_comparable > 0 else None,
                "median_residual": float(np.median(residuals)) if n_comparable > 0 else None,
                "std_residual": float(np.std(residuals)) if n_comparable > 0 else None,
                "mae": float(np.mean(np.abs(residuals))) if n_comparable > 0 else None,
                "rmse": float(np.sqrt(np.mean(residuals**2))) if n_comparable > 0 else None,
                "t_statistic": None,
                "t_p_value": None,
                "wilcoxon_statistic": None,
                "wilcoxon_p_value": None,
                "concept_shift_detected": None,
                "n_comparable_compounds": n_comparable,
                "n_excluded_compounds": n_excluded,
            }

        # One-sample t-test: H₀: mean residual = 0 (no systematic offset)
        t_stat, t_p = ttest_1samp(residuals, 0.0)

        # Wilcoxon signed-rank test: non-parametric H₀: median residual = 0
        # Wilcoxon requires non-zero differences
        nonzero_residuals = residuals[residuals != 0]
        if len(nonzero_residuals) >= 5:
            w_stat, w_p = wilcoxon(nonzero_residuals)
        else:
            w_stat, w_p = None, None

        # Concept shift detected if BOTH tests reject H₀
        if w_p is not None:
            concept_shift = t_p < 0.05 and w_p < 0.05
        else:
            concept_shift = t_p < 0.05

        return {
            "mean_residual": float(np.mean(residuals)),
            "median_residual": float(np.median(residuals)),
            "std_residual": float(np.std(residuals)),
            "mae": float(np.mean(np.abs(residuals))),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "t_statistic": float(t_stat),
            "t_p_value": float(t_p),
            "wilcoxon_statistic": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p_value": float(w_p) if w_p is not None else None,
            "concept_shift_detected": concept_shift,
            "n_comparable_compounds": n_comparable,
            "n_excluded_compounds": n_excluded,
        }

    def target_alignment_details(self) -> pd.DataFrame:
        """Return the full per-compound target alignment DataFrame.

        Each row represents a query compound that has sufficient structural similarity
        to the reference dataset (above min_similarity threshold).

        Returns:
            pd.DataFrame: Columns include id, smiles, query_target, nearest_neighbor_id,
                tanimoto_similarity, neighbor_median_target, target_residual
        """
        return self.alignment_df.copy()

    # ---- Visualization ----

    def plot_target_alignment(self, figsize: tuple[int, int] = (14, 5)) -> None:
        """Plot target alignment diagnostics: residual histogram and predicted-vs-actual scatter.

        Args:
            figsize (tuple[int, int]): Figure size (width, height)
        """
        import matplotlib.pyplot as plt

        if len(self.alignment_df) == 0:
            log.warning("No comparable compounds for target alignment plot")
            return

        residuals = self.alignment_df["target_residual"].values
        query_targets = self.alignment_df["query_target"].values
        neighbor_targets = self.alignment_df["neighbor_median_target"].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: Residual distribution
        ax1.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero (perfect alignment)")
        ax1.axvline(
            x=np.mean(residuals),
            color="darkorange",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {np.mean(residuals):.3f}",
        )
        ax1.set_xlabel("Target Residual (query − neighbor median)")
        ax1.set_ylabel("Count")
        ax1.set_title("Concept Shift: Target Residual Distribution")
        ax1.legend()

        # Annotate with summary stats
        summary = self.concept_shift_summary()
        textstr = (
            f"Mean: {summary['mean_residual']:.3f}\n"
            f"Median: {summary['median_residual']:.3f}\n"
            f"MAE: {summary['mae']:.3f}\n"
            f"t-test p: {summary['t_p_value']:.2e}\n"
            f"Shift: {'YES' if summary['concept_shift_detected'] else 'No'}"
        )
        ax1.text(
            0.98,
            0.98,
            textstr,
            transform=ax1.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Right: Query target vs neighbor median target (should be on diagonal if aligned)
        ax2.scatter(neighbor_targets, query_targets, alpha=0.5, s=20, color="steelblue")
        min_val = min(neighbor_targets.min(), query_targets.min())
        max_val = max(neighbor_targets.max(), query_targets.max())
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect alignment")
        ax2.set_xlabel(f"Reference Neighbor Median ({self.target_column})")
        ax2.set_ylabel(f"Query ({self.target_column})")
        ax2.set_title("Target Value: Query vs Reference Neighbors")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_concept_shift_map(
        self,
        df_all: pd.DataFrame,
        id_column: str | None = None,
        x_col: str = "x",
        y_col: str = "y",
        residual_clip: float = 15.0,
        figsize: tuple[int, int] = (14, 10),
    ) -> None:
        """Plot concept shift on a UMAP projection of chemical space.

        Colors query compounds by their target residual relative to nearest reference
        neighbors. Shows WHERE in chemical space the datasets agree vs disagree.

        Three categories of compounds are shown:
            - **Reference** (small gray dots): the baseline dataset
            - **Query — aligned** (blue-white-red by residual): query compounds with
              comparable reference neighbors, colored by target residual magnitude
            - **Query — no overlap** (gray ×): query compounds too dissimilar to any
              reference compound to assess target alignment

        Args:
            df_all (pd.DataFrame): Combined DataFrame with UMAP coordinates for all
                compounds. Typically from ``model.fp_prox_model().df``.
            id_column (str | None): ID column in df_all to join with alignment results.
                Defaults to ``self.id_column_query``.
            x_col (str): Column name for UMAP x-coordinate (default: "x")
            y_col (str): Column name for UMAP y-coordinate (default: "y")
            residual_clip (float): Clip residual color scale to [-clip, +clip] so that
                extreme outliers don't wash out moderate disagreements (default: 15.0)
            figsize (tuple[int, int]): Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        if id_column is None:
            id_column = self.id_column_query

        # Build lookup sets for reference and query IDs
        ref_ids = set(self.df_reference[self.id_column_reference].values)
        query_ids = set(self.df_query[self.id_column_query].values)

        # Map query IDs to their target residuals
        residual_map = dict(zip(self.alignment_df["id"], self.alignment_df["target_residual"]))

        # Split df_all into reference, query-comparable, query-excluded
        mask_ref = df_all[id_column].isin(ref_ids)
        mask_query = df_all[id_column].isin(query_ids)
        mask_comparable = df_all[id_column].isin(residual_map.keys())

        df_ref = df_all[mask_ref]
        df_query_comparable = df_all[mask_query & mask_comparable]
        df_query_excluded = df_all[mask_query & ~mask_comparable]

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Reference compounds — gray background
        ax.scatter(
            df_ref[x_col],
            df_ref[y_col],
            s=15,
            c="lightgray",
            alpha=0.5,
            zorder=1,
            label=f"Reference ({len(df_ref)})",
        )

        # 2. Excluded query compounds — no comparable reference neighbor
        if len(df_query_excluded) > 0:
            ax.scatter(
                df_query_excluded[x_col],
                df_query_excluded[y_col],
                s=25,
                c="dimgray",
                alpha=0.5,
                marker="x",
                zorder=2,
                label=f"Query — no overlap ({len(df_query_excluded)})",
            )

        # 3. Comparable query compounds — colored by target residual
        if len(df_query_comparable) > 0:
            residuals = df_query_comparable[id_column].map(residual_map).values

            # Clip residuals for color scale so outliers don't wash out the signal
            clipped = np.clip(residuals, -residual_clip, residual_clip)
            norm = TwoSlopeNorm(vmin=-residual_clip, vcenter=0, vmax=residual_clip)

            scatter = ax.scatter(
                df_query_comparable[x_col],
                df_query_comparable[y_col],
                s=40,
                c=clipped,
                cmap="RdBu_r",
                norm=norm,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.3,
                zorder=3,
                label=f"Query — comparable ({len(df_query_comparable)})",
            )
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Target Residual (query − reference neighbor median)")

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Concept Shift Map: Target Alignment in Chemical Space")
        ax.legend(loc="upper left", framealpha=0.8)

        # Summary annotation
        summary = self.concept_shift_summary()
        textstr = (
            f"Median residual: {summary['median_residual']:.2f}\n"
            f"MAE: {summary['mae']:.2f}\n"
            f"Comparable: {summary['n_comparable_compounds']}\n"
            f"Shift detected: {'YES' if summary['concept_shift_detected'] else 'No'}"
        )
        ax.text(
            0.02,
            0.02,
            textstr,
            transform=ax.transAxes,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

        plt.tight_layout()
        plt.show()

    def plot_coverage_curve(self, figsize: tuple[int, int] = (10, 6)) -> None:
        """Plot a coverage curve showing how well the reference dataset covers the query.

        The coverage curve is a cumulative distribution: for each Tanimoto similarity
        threshold τ on the x-axis, the y-axis shows what fraction of query compounds
        have a nearest reference neighbor with similarity ≥ τ.

        Visually:
            - A curve that stays high across all thresholds → well-covered query dataset
            - A curve that drops off quickly → query has many novel compounds
            - The shaded "comparable zone" (above min_similarity) shows which compounds
              can be assessed for target alignment

        Args:
            figsize (tuple[int, int]): Figure size (width, height)
        """
        import matplotlib.pyplot as plt

        sims = np.sort(self._cross_nn_similarities)[::-1]  # Descending
        n = len(sims)
        fractions = np.arange(1, n + 1) / n  # Cumulative fraction
        thresholds = sims  # Each similarity value is a threshold

        fig, ax = plt.subplots(figsize=figsize)

        # Main coverage curve
        ax.plot(thresholds, fractions, color="steelblue", linewidth=2.5, label="Coverage curve")
        ax.fill_between(thresholds, fractions, alpha=0.15, color="steelblue")

        # Shade the "comparable zone" (above min_similarity)
        ax.axvline(
            x=self.min_similarity,
            color="darkorange",
            linestyle="--",
            linewidth=2,
            label=f"min_similarity = {self.min_similarity}",
        )
        ax.axvspan(self.min_similarity, 1.0, alpha=0.08, color="darkorange")

        # Mark key coverage fractions
        for tau, label_fmt in [(0.3, "τ≥0.3"), (0.5, "τ≥0.5"), (0.7, "τ≥0.7")]:
            frac = (self._cross_nn_similarities >= tau).sum() / n
            if frac > 0.01:  # Only annotate if non-trivial
                ax.plot(tau, frac, "o", color="steelblue", markersize=8, zorder=5)
                ax.annotate(
                    f"{label_fmt}: {frac:.0%}",
                    xy=(tau, frac),
                    xytext=(tau + 0.03, frac + 0.05),
                    fontsize=9,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
                )

        # Comparable fraction annotation
        comparable_frac = (self._cross_nn_similarities >= self.min_similarity).sum() / n
        ax.text(
            0.98,
            0.98,
            f"Comparable: {comparable_frac:.0%} of query\n"
            f"Novel: {1 - comparable_frac:.0%} of query\n"
            f"Query compounds: {n}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=10,
        )

        ax.set_xlabel("Tanimoto Similarity Threshold (τ)", fontsize=12)
        ax.set_ylabel("Fraction of Query Covered (NN similarity ≥ τ)", fontsize=12)
        ax.set_title("Coverage Curve: Reference Dataset Coverage of Query Chemical Space", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_similarity_residual_funnel(self, figsize: tuple[int, int] = (12, 7)) -> None:
        """Plot the similarity–residual funnel: target residuals vs Tanimoto similarity.

        This is the key diagnostic for dataset alignment. The intuition:

        - As Tanimoto similarity increases (→ right), target residuals should converge
          toward zero — structurally identical compounds should have similar targets.
        - A well-aligned dataset looks like a **funnel narrowing toward zero**.
        - Concept shift shows up as:
            - **Offset funnel**: narrows but centered away from zero → systematic bias
            - **Wide funnel**: doesn't narrow → noisy/incompatible data
            - **Outlier wings**: specific compounds with large disagreements

        A running median line and IQR band highlight the trend, while individual
        points show per-compound detail colored by residual magnitude.

        Args:
            figsize (tuple[int, int]): Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        if len(self.alignment_df) == 0:
            log.warning("No comparable compounds for similarity-residual funnel plot")
            return

        sims = self.alignment_df["tanimoto_similarity"].values
        residuals = self.alignment_df["target_residual"].values

        fig, ax = plt.subplots(figsize=figsize)

        # Color points by residual magnitude (diverging colormap centered at 0)
        abs_max = max(abs(residuals.min()), abs(residuals.max()), 1e-6)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        scatter = ax.scatter(
            sims,
            residuals,
            c=residuals,
            cmap="RdBu_r",
            norm=norm,
            alpha=0.5,
            s=25,
            edgecolors="none",
            zorder=2,
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Target Residual (query − reference neighbor median)")

        # Zero line (perfect alignment)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.7, zorder=1)

        # Running median + IQR band using similarity bins
        bin_edges = np.linspace(sims.min(), sims.max(), 15)
        bin_centers = []
        medians = []
        q25s = []
        q75s = []

        for i in range(len(bin_edges) - 1):
            mask = (sims >= bin_edges[i]) & (sims < bin_edges[i + 1])
            if mask.sum() >= 5:  # Need enough points for meaningful statistics
                bin_residuals = residuals[mask]
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                medians.append(np.median(bin_residuals))
                q25s.append(np.percentile(bin_residuals, 25))
                q75s.append(np.percentile(bin_residuals, 75))

        if len(bin_centers) >= 3:
            bin_centers = np.array(bin_centers)
            medians = np.array(medians)
            q25s = np.array(q25s)
            q75s = np.array(q75s)

            ax.plot(
                bin_centers,
                medians,
                color="darkorange",
                linewidth=2.5,
                zorder=4,
                label="Running median",
            )
            ax.fill_between(
                bin_centers,
                q25s,
                q75s,
                alpha=0.2,
                color="darkorange",
                zorder=3,
                label="IQR (25th–75th percentile)",
            )

        # Min similarity threshold line
        ax.axvline(
            x=self.min_similarity,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"min_similarity = {self.min_similarity}",
        )

        # Summary annotation
        summary = self.concept_shift_summary()
        funnel_diagnosis = self._diagnose_funnel(sims, residuals)
        textstr = (
            f"Median residual: {summary['median_residual']:.3f}\n"
            f"MAE: {summary['mae']:.3f}\n"
            f"Shift detected: {'YES' if summary['concept_shift_detected'] else 'No'}\n"
            f"Pattern: {funnel_diagnosis}"
        )
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=10,
        )

        ax.set_xlabel("Tanimoto Similarity (query → reference NN)", fontsize=12)
        ax.set_ylabel("Target Residual (query − reference neighbor median)", fontsize=12)
        ax.set_title("Similarity–Residual Funnel: Target Agreement vs Chemical Similarity", fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _diagnose_funnel(sims: np.ndarray, residuals: np.ndarray) -> str:
        """Diagnose the funnel pattern from similarity-residual data.

        Args:
            sims (np.ndarray): Tanimoto similarities
            residuals (np.ndarray): Target residuals

        Returns:
            str: Human-readable diagnosis of the funnel pattern
        """
        # Split into low-similarity and high-similarity halves
        median_sim = np.median(sims)
        low_mask = sims < median_sim
        high_mask = sims >= median_sim

        low_iqr = (
            np.percentile(residuals[low_mask], 75) - np.percentile(residuals[low_mask], 25)
            if low_mask.sum() > 5
            else float("inf")
        )
        high_iqr = (
            np.percentile(residuals[high_mask], 75) - np.percentile(residuals[high_mask], 25)
            if high_mask.sum() > 5
            else float("inf")
        )

        narrows = high_iqr < low_iqr * 0.7  # IQR narrows by at least 30%
        offset = abs(np.median(residuals)) > 0.5 * np.std(residuals)

        if narrows and not offset:
            return "Aligned funnel ✓"
        elif narrows and offset:
            return "Offset funnel (systematic bias)"
        elif not narrows and not offset:
            return "Wide funnel (noisy data)"
        else:
            return "Wide + offset (incompatible)"

    def plot_covariate_shift(self, bins: int = 50, figsize: tuple[int, int] = (10, 6)) -> None:
        """Plot overlaid histograms of within-reference and cross-dataset NN similarity distributions.

        Args:
            bins (int): Number of histogram bins
            figsize (tuple[int, int]): Figure size (width, height)
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            self._ref_nn_similarities,
            bins=bins,
            alpha=0.5,
            label="Reference (within-dataset)",
            edgecolor="black",
            color="steelblue",
        )
        ax.hist(
            self._cross_nn_similarities,
            bins=bins,
            alpha=0.5,
            label="Query (cross-dataset)",
            edgecolor="black",
            color="darkorange",
        )

        ax.set_xlabel("Nearest Neighbor Tanimoto Similarity")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Covariate Shift: {len(self._ref_nn_similarities)} ref vs "
            f"{len(self._cross_nn_similarities)} query compounds"
        )
        ax.legend()

        summary = self.covariate_shift_summary()
        textstr = (
            f"KS stat: {summary['ks_statistic']:.3f} (p={summary['ks_p_value']:.2e})\n"
            f"JSD: {summary['jensen_shannon_divergence']:.4f}\n"
            f"PSI: {summary['population_stability_index']:.4f} ({summary['psi_severity']})"
        )
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
    from workbench.utils.test_data_generator import TestDataGenerator

    test_data = TestDataGenerator()

    # Test all 9 combinations of overlap × alignment
    for overlap in ["high", "medium", "low"]:
        for alignment_level in ["high", "medium", "low"]:
            print("=" * 80)
            print(f"Testing: overlap={overlap}, alignment={alignment_level}")
            print("=" * 80)

            ref_df, query_df = test_data.aqsol_alignment_data(overlap=overlap, alignment=alignment_level)
            print(f"Reference: {len(ref_df)}, Query: {len(query_df)}")

            da = DatasetAlignment(
                ref_df,
                query_df,
                target_column="solubility",
                id_column_reference="id",
                id_column_query="id",
            )

            # Summaries
            print("\n--- Covariate Shift ---")
            cov = da.covariate_shift_summary()
            for k, v in cov.items():
                print(f"  {k}: {v}")

            print("\n--- Concept Shift ---")
            cs = da.concept_shift_summary()
            for k, v in cs.items():
                print(f"  {k}: {v}")

            # New plots
            da.plot_coverage_curve()
            da.plot_similarity_residual_funnel()

            print()

    print("=" * 80)
    print("All DatasetAlignment tests completed!")
    print("=" * 80)
