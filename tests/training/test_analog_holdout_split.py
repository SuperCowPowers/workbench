"""Tests for the analog-expansion holdout split."""

import numpy as np
import pandas as pd
import pytest

from workbench.training.splits import analog_holdout_split
from workbench.utils.chem_utils.fingerprints import similarity_fingerprints

# Three tight series (benzene, biphenyl-ish, and a long alkane chain) so nearest
# neighbors are unambiguous, plus potency assigned per series
SERIES = {
    "phenol": ["c1ccccc1O", "Cc1ccccc1O", "CCc1ccccc1O", "CCCc1ccccc1O"],
    "aniline": ["c1ccccc1N", "Cc1ccccc1N", "CCc1ccccc1N", "CCCc1ccccc1N"],
    "alkane": ["CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC"],
}


def _frame() -> pd.DataFrame:
    rows = []
    # phenols are the potent series, anilines mid, alkanes weak
    for potency, (series, smiles) in zip([8.0, 6.0, 4.0], SERIES.items()):
        for i, smi in enumerate(smiles):
            rows.append({"smiles": smi, "series": series, "pic50": potency - i * 0.1})
    return pd.DataFrame(rows)


class TestSimilarityFingerprints:
    def test_returns_positions_for_parseable_only(self):
        fps, positions = similarity_fingerprints(["c1ccccc1", "not-a-molecule", "CCO"])
        assert len(fps) == 2
        assert positions == [0, 2]

    def test_handles_nan_and_empty(self):
        fps, positions = similarity_fingerprints(["c1ccccc1", np.nan, "", None])
        assert len(fps) == 1
        assert positions == [0]

    def test_defaults_to_sparse_counts(self):
        (fp,), _ = similarity_fingerprints(["CCCCc1ccccc1O"])
        assert "SparseIntVect" in type(fp).__name__

    def test_folding_and_binary_are_opt_in(self):
        (folded,), _ = similarity_fingerprints(["CCCCc1ccccc1O"], fp_size=2048)
        (binary,), _ = similarity_fingerprints(["CCCCc1ccccc1O"], counts=False, fp_size=2048)
        assert type(folded).__name__ == "UIntSparseIntVect"
        assert type(binary).__name__ == "ExplicitBitVect"

    def test_counts_score_analog_pairs_higher_than_binary(self):
        """Multiset Tanimoto keeps substructure frequency, so homologs score closer."""
        from rdkit import DataStructs

        pair = ["CCCCc1ccccc1O", "CCc1ccccc1O"]
        (c1, c2), _ = similarity_fingerprints(pair)
        (b1, b2), _ = similarity_fingerprints(pair, counts=False, fp_size=2048)
        assert DataStructs.TanimotoSimilarity(c1, c2) > DataStructs.TanimotoSimilarity(b1, b2)


class TestAnalogHoldoutSplit:
    def test_holds_out_the_potent_series(self):
        df = _frame()
        holdout = analog_holdout_split(df, "pic50", n_hits=1, analogs_per_hit=3, min_similarity=0.3)
        held = df[holdout]["series"].unique()
        assert "phenol" in held

    def test_holdout_is_a_boolean_mask_over_rows(self):
        df = _frame()
        holdout = analog_holdout_split(df, "pic50", n_hits=1, analogs_per_hit=2)
        assert holdout.dtype == bool
        assert len(holdout) == len(df)

    def test_hit_count_bounds_the_holdout(self):
        """Each hit contributes at most analogs_per_hit neighbors, and never itself."""
        df = _frame()
        holdout = analog_holdout_split(df, "pic50", n_hits=2, analogs_per_hit=2, min_similarity=0.0)
        assert holdout.sum() <= 2 * 2

    def test_hits_stay_in_training(self):
        """The most potent compounds are the expanded hits — they belong in training."""
        df = _frame()
        holdout = analog_holdout_split(df, "pic50", n_hits=3, analogs_per_hit=3, min_similarity=0.0)
        top_three = df["pic50"].nlargest(3).index
        assert not holdout[top_three].any()

    def test_holdout_is_not_potency_enriched_by_construction(self):
        """Excluding hits is what keeps the eval set from being biased upward."""
        df = _frame()
        holdout = analog_holdout_split(df, "pic50", n_hits=2, analogs_per_hit=3, min_similarity=0.0)
        assert df[holdout]["pic50"].max() < df[~holdout]["pic50"].max()

    def test_more_analogs_per_hit_grows_the_holdout(self):
        df = _frame()
        small = analog_holdout_split(df, "pic50", n_hits=1, analogs_per_hit=1, min_similarity=0.0)
        large = analog_holdout_split(df, "pic50", n_hits=1, analogs_per_hit=3, min_similarity=0.0)
        assert large.sum() > small.sum()

    def test_high_similarity_cutoff_empties_the_holdout(self):
        """Nothing clears a cutoff of 1.0, and hits are never held out themselves."""
        df = _frame()
        holdout = analog_holdout_split(df, "pic50", n_hits=2, analogs_per_hit=5, min_similarity=1.0)
        assert holdout.sum() == 0

    def test_multiple_targets_union_their_holdouts(self):
        df = _frame()
        # a second target that ranks the alkane series highest
        df["other"] = np.where(df["series"] == "alkane", 9.0, 1.0)
        one = analog_holdout_split(df, "pic50", n_hits=1, analogs_per_hit=2, min_similarity=0.3)
        both = analog_holdout_split(df, ["pic50", "other"], n_hits=1, analogs_per_hit=2, min_similarity=0.3)
        assert both.sum() > one.sum()
        assert set(df[one].index) <= set(df[both].index)

    def test_nan_targets_are_never_hits(self):
        df = _frame()
        df.loc[df["series"] == "phenol", "pic50"] = np.nan
        holdout = analog_holdout_split(df, "pic50", n_hits=1, analogs_per_hit=0, min_similarity=1.0)
        assert not df[holdout]["pic50"].isna().any()

    def test_unparseable_smiles_are_excluded(self):
        df = _frame()
        df.loc[len(df)] = {"smiles": "not-a-molecule", "series": "junk", "pic50": 99.0}
        holdout = analog_holdout_split(df, "pic50", n_hits=1, analogs_per_hit=2, min_similarity=0.3)
        assert not holdout[df["series"] == "junk"].any()

    def test_missing_target_column_raises(self):
        with pytest.raises(ValueError, match="Target column"):
            analog_holdout_split(_frame(), "nope")

    def test_missing_smiles_column_raises(self):
        with pytest.raises(ValueError, match="SMILES column"):
            analog_holdout_split(_frame().drop(columns=["smiles"]), "pic50")
