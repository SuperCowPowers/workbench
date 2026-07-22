"""Multi-task alignment: chemical-space coverage and per-aux concordance against a primary target.

Driver: deciding whether a multi-task chemprop run (e.g. ``mdr1_er + caco2_er + caco2_pappab + logd``
auxiliaries) will lift over a single-task model on the primary. Two ingredients matter:

    1. **Chemical-space coverage** — do auxiliary compounds occupy the same chemistry as the
       primary? Strong coverage means the aux head's gradient on shared chemistry can refine
       the encoder. Aux-only chemistry that's well-connected to primary extends coverage.
    2. **Per-aux alignment** — where they overlap, do the targets agree (Pearson r) and do
       the local SAR neighborhoods predict each other (z-scored residual)?

Build one shared ``FingerprintProximity`` (ECFP + KNN + UMAP) on the union of all rows that
have any target value, then per aux compute label-only stats (counts, Pearson r, recommendation)
and chemistry stats (Tanimoto coverage, z-scored residual).

For the matching UI / "map" view, see
``workbench.web_interface.components.plugins.MultiTaskAlignmentMap``.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

log = logging.getLogger("workbench")


class MultiTaskAlignment:
    """Per-aux alignment of a multi-task DataFrame against a primary target.

    Input is a wide multi-task DataFrame: one row per compound, with ``id_column``,
    ``smiles``, the primary target column, and one or more auxiliary target columns.
    Targets are NaN where not measured.

    A single ``FingerprintProximity`` is built on the union of all rows that have any
    target value, so every per-aux computation reuses the same fingerprints, KNN graph,
    and UMAP coordinates.

    Use ``summary()`` for the per-aux quantitative table, ``results()`` for the
    per-compound DataFrame (with shared UMAP coords + per-aux ``tanimoto_to_primary_<aux>``
    and ``residual_<aux>`` columns), and ``neighbors()`` for compound-level lookups.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        primary: str,
        auxiliaries: Optional[list[str]] = None,
        id_column: str = "id",
        k_neighbors: int = 5,
        radius: int = 2,
        n_bits: int = 4096,
        min_n_shared: int = 10,
        extension_ratio_threshold: float = 0.5,
    ) -> None:
        """Initialize the alignment and run all per-aux computations.

        Args:
            df: Wide multi-task DataFrame with ``id_column``, ``smiles``, ``primary``,
                and aux target columns. NaN means the target wasn't measured.
            primary: Name of the primary target column.
            auxiliaries: Aux target column names. If None, defaults to every numeric
                column that isn't ``id_column``, ``smiles``, or ``primary``.
            id_column: Identifier column name (default: ``"id"``).
            k_neighbors: Number of primary-having neighbors used for the residual
                computation (default: 5).
            radius: Morgan fingerprint radius (default: 2 = ECFP4).
            n_bits: Number of fingerprint bits (default: 4096).
            min_n_shared: Minimum rows-with-both-targets before Pearson r is trusted
                (default: 10).
            extension_ratio_threshold: ``n_aux_only / n_primary`` above which the
                extension region counts as "substantial volume" (default: 0.5).
        """
        for col in (id_column, "smiles", primary):
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col!r}")

        if auxiliaries is None:
            reserved = {id_column, "smiles", primary}
            auxiliaries = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
        else:
            for aux in auxiliaries:
                if aux not in df.columns:
                    raise ValueError(f"Aux column {aux!r} not in DataFrame")
                if aux == primary:
                    raise ValueError(f"Aux {aux!r} is the same as primary")

        if not auxiliaries:
            raise ValueError("No auxiliaries provided and none auto-detected from numeric columns")

        self.id_column = id_column
        self.primary = primary
        self.auxiliaries = list(auxiliaries)
        self.k_neighbors = k_neighbors
        self.min_n_shared = min_n_shared
        self.extension_ratio_threshold = extension_ratio_threshold

        all_targets = [primary, *self.auxiliaries]

        # Drop rows with no target values at all — they can't contribute
        any_target = df[all_targets].notna().any(axis=1)
        n_dropped = int((~any_target).sum())
        if n_dropped:
            log.info(f"Dropping {n_dropped} rows with no target values")
        df = df.loc[any_target].copy()

        dup_mask = df.duplicated(subset=id_column, keep="first")
        if dup_mask.any():
            log.warning(f"Dropping {dup_mask.sum()} duplicate {id_column!r} value(s) (keeping first)")
            df = df.loc[~dup_mask].copy()

        log.info(f"MultiTaskAlignment: {len(df)} compounds, primary={primary!r}, " f"auxiliaries={self.auxiliaries}")

        self._prox = FingerprintProximity(
            df,
            id_column=id_column,
            target=None,
            include_all_columns=True,
            radius=radius,
            n_bits=n_bits,
        )

        log.info("Computing per-compound alignment metrics...")
        self._per_compound = self._compute_per_compound()
        self._summary = self._compute_summary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def results(self) -> pd.DataFrame:
        """Per-compound DataFrame with shared UMAP coords and per-aux alignment columns.

        Returns:
            DataFrame with original columns (``id``, ``smiles``, primary, all auxes), plus:
                - ``x``, ``y``: shared UMAP 2D coordinates
                - ``tanimoto_to_primary``: best Tanimoto similarity from this row to any
                  primary-having row (1.0 for primary rows themselves)
                - ``residual_<aux>``: z-scored residual for each aux, defined only on
                  rows where the aux is measured. Computed as
                  ``z(aux) - median(z(primary)`` over top-k primary-having neighbors``)``.
                  Sign indicates direction of disagreement; magnitude is in std units.
        """
        df = self._prox.df.copy()
        internal_cols = [
            "nn_distance",
            "nn_id",
            "nn_target",
            "nn_target_diff",
            "nn_similarity",
            "fingerprint",
        ]
        df = df.drop(columns=[c for c in internal_cols if c in df.columns])
        df = df.merge(self._per_compound, on=self.id_column, how="left")
        return df

    def summary(self) -> pd.DataFrame:
        """Per-aux quantitative summary, one row per aux.

        Returns:
            DataFrame with columns:
                - ``aux``: aux target name
                - ``n_primary``, ``n_aux``, ``n_shared``, ``n_aux_only``: row counts
                - ``pearson_r``: correlation on shared rows (NaN if ``n_shared < min_n_shared``)
                - ``r_confidence``: ``high`` / ``moderate`` / ``low`` / ``unmeasured``
                - ``tanimoto_coverage_mean``: mean Tanimoto from aux-having rows to nearest
                  primary-having row
                - ``frac_coverage_ge_05``, ``frac_coverage_ge_03``: fraction of aux-having
                  rows with Tanimoto coverage above the threshold
                - ``residual_abs_mean``, ``residual_abs_p95``: z-scored residual stats over
                  aux-having rows that have at least one primary neighbor
                - ``overlap``: ``Beneficial`` / ``Neutral`` / ``Harmful`` / ``N/A``
                - ``extension``: ``Strong`` / ``Modest`` / ``Minimal`` / ``None``
                - ``recommendation``: ``Use`` / ``Marginal`` / ``Risky`` / ``Skip``
        """
        return self._summary.copy()

    def neighbors(
        self,
        compound_id: Union[str, int, list],
        n_neighbors: int = 10,
    ) -> pd.DataFrame:
        """Nearest neighbors for a compound (or list) in the shared fingerprint space.

        Args:
            compound_id: ID or list of IDs to look up.
            n_neighbors: Number of neighbors to return (default: 10).

        Returns:
            DataFrame sorted by similarity (descending) with the query id, ``neighbor_id``,
            ``similarity``, ``smiles``, and the primary + aux target columns.
        """
        nbrs = self._prox.neighbors(compound_id, n_neighbors=n_neighbors)
        keep = [self.id_column, "neighbor_id", "similarity", "smiles", self.primary, *self.auxiliaries]
        return nbrs[[c for c in keep if c in nbrs.columns]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_per_compound(self) -> pd.DataFrame:
        """Per-row chemistry signals: Tanimoto coverage and z-scored residual per aux."""
        df = self._prox.df
        all_ids = df[self.id_column].tolist()
        primary_mask = df[self.primary].notna()
        primary_ids = set(df.loc[primary_mask, self.id_column])

        # One bulk neighbor lookup; we'll filter to primary-having neighbors below
        n_lookup = max(50, self.k_neighbors * 10)
        nbrs = self._prox.neighbors(all_ids, n_neighbors=n_lookup)

        # Drop self-neighbors so a primary row's own value doesn't satisfy its coverage
        nbrs_no_self = nbrs[nbrs[self.id_column] != nbrs["neighbor_id"]].copy()
        primary_nbrs = nbrs_no_self[nbrs_no_self["neighbor_id"].isin(primary_ids)].copy()

        # Tanimoto-to-primary: best similarity to any primary-having compound
        best_sim = primary_nbrs.groupby(self.id_column)["similarity"].max()

        result = pd.DataFrame({self.id_column: all_ids})
        result["tanimoto_to_primary"] = result[self.id_column].map(best_sim).fillna(0.0)
        # Primary rows are by definition fully covered
        result.loc[result[self.id_column].isin(primary_ids), "tanimoto_to_primary"] = 1.0

        # Z-scored "predicted primary" from top-k primary-having neighbors
        primary_nbrs = primary_nbrs.sort_values([self.id_column, "similarity"], ascending=[True, False])
        primary_nbrs["_rank"] = primary_nbrs.groupby(self.id_column).cumcount() + 1
        topk = primary_nbrs[primary_nbrs["_rank"] <= self.k_neighbors].copy()

        primary_vals = df.set_index(self.id_column)[self.primary]
        primary_z = self._zscore(primary_vals)
        topk["primary_z"] = topk["neighbor_id"].map(primary_z)
        primary_z_pred = topk.groupby(self.id_column)["primary_z"].median()

        # Per-aux z-scored residual: defined only where the aux is measured
        for aux in self.auxiliaries:
            aux_vals = df.set_index(self.id_column)[aux]
            aux_z = self._zscore(aux_vals)
            aux_z_aligned = result[self.id_column].map(aux_z)
            primary_pred_aligned = result[self.id_column].map(primary_z_pred)
            residual = aux_z_aligned - primary_pred_aligned
            # Mask rows that don't have the aux value
            aux_present = result[self.id_column].map(aux_vals.notna()).fillna(False).values
            result[f"residual_{aux}"] = np.where(aux_present, residual.values, np.nan)

        return result

    def _compute_summary(self) -> pd.DataFrame:
        """Per-aux quantitative summary (counts, pearson, coverage, residuals, verdicts)."""
        df = self._prox.df
        primary_mask = df[self.primary].notna()
        n_primary = int(primary_mask.sum())
        rows = []

        for aux in self.auxiliaries:
            aux_mask = df[aux].notna()
            n_aux = int(aux_mask.sum())
            n_shared = int((primary_mask & aux_mask).sum())
            n_aux_only = int((~primary_mask & aux_mask).sum())

            if n_shared >= self.min_n_shared:
                pearson_r = float(df.loc[primary_mask & aux_mask, [self.primary, aux]].corr().iloc[0, 1])
            else:
                pearson_r = float("nan")

            aux_ids = df.loc[aux_mask, self.id_column]
            cov = self._per_compound.set_index(self.id_column).loc[aux_ids, "tanimoto_to_primary"]
            cov_mean = float(cov.mean()) if len(cov) else 0.0
            cov_05 = float((cov >= 0.5).mean()) if len(cov) else 0.0
            cov_03 = float((cov >= 0.3).mean()) if len(cov) else 0.0

            residuals = self._per_compound[f"residual_{aux}"].dropna()
            res_abs_mean = float(residuals.abs().mean()) if len(residuals) else float("nan")
            res_abs_p95 = float(residuals.abs().quantile(0.95)) if len(residuals) else float("nan")

            overlap, _ = _assess_overlap(pearson_r, n_shared, self.min_n_shared)
            extension, _ = _assess_extension(pearson_r, n_aux_only, n_primary, self.extension_ratio_threshold)
            recommendation, _ = _combine_assessments(overlap, extension)

            rows.append(
                {
                    "aux": aux,
                    "n_primary": n_primary,
                    "n_aux": n_aux,
                    "n_shared": n_shared,
                    "n_aux_only": n_aux_only,
                    "pearson_r": pearson_r,
                    "r_confidence": _confidence_tier(n_shared, self.min_n_shared),
                    "tanimoto_coverage_mean": cov_mean,
                    "frac_coverage_ge_05": cov_05,
                    "frac_coverage_ge_03": cov_03,
                    "residual_abs_mean": res_abs_mean,
                    "residual_abs_p95": res_abs_p95,
                    "overlap": overlap,
                    "extension": extension,
                    "recommendation": recommendation,
                }
            )
            r_str = f"r={pearson_r:.3f}" if not np.isnan(pearson_r) else "r=NA"
            log.info(
                f"  {aux}: shared={n_shared:,} aux_only={n_aux_only:,} {r_str} "
                f"cov_mean={cov_mean:.2f} -> overlap={overlap}, extension={extension} "
                f"-> {recommendation}"
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        """Z-score a Series; returns zeros if std is zero (constant column)."""
        std = s.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return s - s.mean()
        return (s - s.mean()) / std


# ----------------------------------------------------------------------
# Verdict helpers (label-only scoring; same thresholds across multi-task code)
# ----------------------------------------------------------------------


def _confidence_tier(n_shared: int, min_n_shared: int) -> str:
    """Bucket how trustworthy a Pearson r is, given shared-compound count."""
    if n_shared < min_n_shared:
        return "unmeasured"
    if n_shared < 30:
        return "low"
    if n_shared < 100:
        return "moderate"
    return "high"


def _assess_overlap(r: float, n_shared: int, min_n_shared: int) -> tuple[str, str]:
    """Score the overlap region (compounds with both primary and aux measured).

    Thresholds (label-correlation on shared rows):
        r in [0.4, 0.95]  -> Beneficial : sweet spot — heads predict related but distinct targets
        r > 0.95          -> Neutral    : redundant; aux head just re-weights primary
        r < 0.4           -> Harmful    : discordant; gradient conflict / negative-transfer risk
        n_shared too low  -> N/A
    """
    if n_shared < min_n_shared:
        return ("N/A", f"only {n_shared} shared compounds (need >= {min_n_shared} to score)")
    if 0.4 <= r <= 0.95:
        return (
            "Beneficial",
            f"sweet-spot r={r:.2f} on {n_shared:,} shared compounds — encoder learns richer features",
        )
    if r > 0.95:
        return (
            "Neutral",
            f"redundant r={r:.2f} on {n_shared:,} shared compounds — aux head just re-weights primary",
        )
    return (
        "Harmful",
        f"discordant r={r:.2f} on {n_shared:,} shared compounds — gradient conflict, negative-transfer risk",
    )


def _assess_extension(
    r: float,
    n_aux_only: int,
    n_primary: int,
    ratio_threshold: float,
) -> tuple[str, str]:
    """Score the extension region (aux-only compounds; primary head is masked there)."""
    if n_aux_only == 0:
        return ("None", "no aux-only compounds")

    ratio = n_aux_only / n_primary if n_primary > 0 else 0.0
    has_volume = ratio >= ratio_threshold

    if np.isnan(r):
        sim_str = "task similarity unknown (no overlap to measure)"
        if has_volume:
            return ("Strong", f"{ratio:.1f}x primary of novel chemistry; {sim_str}")
        return ("Modest", f"{n_aux_only:,} aux-only compounds ({ratio:.1f}x primary); {sim_str}")

    similar = r >= 0.4
    if has_volume and similar:
        return ("Strong", f"{ratio:.1f}x primary of novel chemistry x similar task (r={r:.2f})")
    if has_volume:
        return ("Modest", f"{ratio:.1f}x primary of novel chemistry but weak similarity (r={r:.2f})")
    if similar:
        return ("Modest", f"limited volume ({ratio:.1f}x primary) but similar task (r={r:.2f})")
    return ("Minimal", f"low volume ({ratio:.1f}x primary) and weak similarity (r={r:.2f})")


def _combine_assessments(overlap: str, extension: str) -> tuple[str, str]:
    """Combine overlap + extension scores into an actionable recommendation."""
    if overlap == "Harmful":
        if extension in ("Strong", "Modest"):
            return ("Risky", "harmful overlap; extension might rescue but cross-task signal could hurt primary")
        return ("Skip", "negative transfer from overlap with no extension to compensate")

    if overlap == "Beneficial":
        if extension in ("Strong", "Modest"):
            return ("Use", "both mechanisms contribute lift")
        return ("Use", "cross-task signal contributes lift")

    # overlap is Neutral or N/A — extension is the only available mechanism
    if extension == "Strong":
        return ("Use", "extension is the primary lift mechanism")
    if extension == "Modest":
        return ("Marginal", "limited extension lift; consider domain knowledge")
    return ("Skip", "no clear lift mechanism")


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    from workbench.utils.synthetic_data_generator import SyntheticDataGenerator

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1400)

    # Build a synthetic multi-task DataFrame from two AQSol partitions
    ref_df, query_df = SyntheticDataGenerator().aqsol_alignment_data(overlap="medium", alignment="medium")
    ref_df = ref_df.assign(id=ref_df["id"].astype(str).radd("ref_"))
    query_df = query_df.assign(id=query_df["id"].astype(str).radd("qry_"))
    ref_df = ref_df.rename(columns={"solubility": "primary_sol"})
    query_df = query_df.rename(columns={"solubility": "aux_sol"})
    mt_df = pd.concat(
        [
            ref_df[["id", "smiles", "primary_sol"]],
            query_df[["id", "smiles", "aux_sol"]],
        ],
        ignore_index=True,
    )

    mta = MultiTaskAlignment(mt_df, primary="primary_sol", auxiliaries=["aux_sol"], id_column="id")

    print("\n=== summary() ===")
    print(mta.summary().to_string(index=False))

    print("\n=== results() (first 5 rows) ===")
    print(mta.results().head().to_string(index=False))

    print("\nMultiTaskAlignment tests completed!")
