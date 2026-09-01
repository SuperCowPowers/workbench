"""Does the Tox21 -> challenge pIC50 offset hold below pIC50 5.0?

Pooling Tox21's CYP2D6 potency into the scored column needs one number: the offset
between the two assays. It is measured on compounds both assays ran -- 119 of them, only
20 below pIC50 5.0 -- and applied to 1,528 compounds, two thirds of which sit below 5.0.
So the offset is measured on potent compounds and spent on weak ones.

This extends the anchor set with high-Tanimoto neighbour pairs to get counts down there,
and it treats that expansion as the suspect it is. A neighbour pair carries the cross-assay
offset plus a structure-activity difference, and that second term biases any band table
built by conditioning on one side of the pair: pick the compounds whose *challenge* value
is low and their neighbours' Tox21 values regress toward the Tox21 mean, which reads as
the offset drifting negative. Three guards:

  reach       how similar the nearest Tox21 compound actually is. An anchor set that
              cannot be extended is the answer to the question, not a failure to run it.
  symmetry    every band table is built twice, binning on the Tox21 value and on the
              challenge value. Errors on both axes move those two in opposite directions;
              a real band-dependent offset moves them the same way.
  floor       Tox21 fits a curve for a fifth of what it screens, so what pooling can
              supply is bounded by that floor rather than by the offset.

Binning on the Tox21 value is the primary read, because the correction is applied given a
Tox21 reading -- which also makes the conditional mean, not the structural slope, the
estimator the pooling step wants.

    python cyp_tox21_anchors.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

RDLogger.DisableLog("rdApp.*")

DATA = "../../../../data/public_data/output/comp_chem"
CHALLENGE = f"{DATA}/openadmet/cyp/training/inhibition.csv"
TOX21 = f"{DATA}/tox21/cyp_inhibition/cyp2d6.csv"

TARGET = "cyp2d6_pic50_direct_inhibition"
SIM_THRESHOLDS = [0.7, 0.6, 0.5]
BANDS = [(0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 99)]


def skeletons(smiles: pd.Series) -> pd.Series:
    """InChIKey connectivity block — matches structures across salt and stereo variants."""
    keys = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        keys.append(Chem.MolToInchiKey(mol).split("-")[0] if mol else None)
    return pd.Series(keys, index=smiles.index)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Challenge and Tox21 CYP2D6 potency, keyed on skeleton."""
    ch = pd.read_csv(CHALLENGE)[["molecule_name", "smiles", TARGET]].dropna(subset=[TARGET])
    ch["key"] = skeletons(ch["smiles"])
    ch = ch.dropna(subset=["key"]).drop_duplicates(subset=["key"]).rename(columns={TARGET: "challenge"})

    tox = pd.read_csv(TOX21)
    tox = tox[~tox["luciferase_inhibitor"].astype(bool)]
    tox = tox[["smiles", "pic50", "pic50_sd"]].dropna(subset=["smiles", "pic50"])
    tox["key"] = skeletons(tox["smiles"])
    tox = tox.dropna(subset=["key"])
    # Duplicate skeletons are the same structure run as separate samples; median them.
    tox = tox.groupby("key", as_index=False).agg(smiles=("smiles", "first"), tox21=("pic50", "median"))
    return ch, tox


def band_table(pairs: pd.DataFrame, on: str) -> pd.DataFrame:
    """Offset by band, binning on `on` ('tox21' or 'challenge')."""
    rows = []
    for lo, hi in BANDS:
        sel = pairs[(pairs[on] >= lo) & (pairs[on] < hi)]
        rows.append(
            {
                "band": f"{lo}-{hi}" if hi < 99 else f">{lo}",
                "n": len(sel),
                "offset": sel["delta"].mean() if len(sel) else np.nan,
                "sd": sel["delta"].std() if len(sel) > 1 else np.nan,
                "sem": sel["delta"].std() / np.sqrt(len(sel)) if len(sel) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def report(name: str, pairs: pd.DataFrame) -> None:
    print(f"\n--- {name}: n={len(pairs)}, offset={pairs['delta'].mean():+.3f}, sd={pairs['delta'].std():.3f}")
    for on in ("tox21", "challenge"):
        t = band_table(pairs, on)
        line = "  ".join(
            f"{r.band}: {r.offset:+.2f}({int(r.n)})" if r.n else f"{r.band}: --(0)" for r in t.itertuples()
        )
        print(f"    by {on:9s} {line}")


def main():
    ch, tox = load()
    print(f"challenge CYP2D6 {len(ch):,} compounds, tox21 CYP2D6 {len(tox):,} with a fitted pIC50")

    # --- the measured anchors: same structure, both assays -----------------------------
    exact = ch.merge(tox[["key", "tox21"]], on="key")
    exact["delta"] = exact["challenge"] - exact["tox21"]
    report("exact overlap", exact)
    below = exact[exact["tox21"] < 5.0]
    print(f"    below 5.0 (tox21 side): n={len(below)}, offset={below['delta'].mean():+.3f}")

    # --- Tox21's own floor: a truncated source drifts without any scale change ---------
    q = tox["tox21"].quantile([0.0, 0.01, 0.05, 0.25, 0.5]).round(2).to_dict()
    print(f"\ntox21 pIC50 floor: min {q[0.0]}, p01 {q[0.01]}, p05 {q[0.05]}, p25 {q[0.25]}, median {q[0.5]}")
    print(f"    challenge pIC50 min {ch['challenge'].min():.2f}, p05 {ch['challenge'].quantile(0.05):.2f}")

    # --- can the anchor set be extended at all? ----------------------------------------
    prox = FingerprintProximity(tox.rename(columns={"key": "id"}), id_column="id", target="tox21")
    query = ch[~ch["key"].isin(set(tox["key"]))][["key", "smiles", "challenge"]]
    query = query.rename(columns={"key": "query_id"})
    nearest = prox.neighbors_from_query_df(query[["query_id", "smiles"]], n_neighbors=1)
    print(f"\n{len(query):,} challenge compounds have no exact Tox21 match. Similarity to the")
    print("nearest Tox21 compound carrying a fitted pIC50:")
    q = nearest["similarity"].quantile([0.5, 0.9, 0.99]).round(3).to_dict()
    print(f"    median {q[0.5]}, p90 {q[0.9]}, p99 {q[0.99]}, max {nearest['similarity'].max():.3f}")
    print("    " + "  ".join(f">={t}: {(nearest['similarity'] >= t).sum()}" for t in SIM_THRESHOLDS))

    for thresh in SIM_THRESHOLDS:
        nn = nearest[nearest["similarity"] >= thresh]
        if len(nn) < 10:
            print(f"\n--- Tanimoto >= {thresh}: {len(nn)} pairs, too few to anchor anything")
            continue
        pairs = nn.merge(query[["query_id", "challenge"]], on="query_id")
        pairs["delta"] = pairs["challenge"] - pairs["tox21"]
        report(f"Tanimoto >= {thresh}", pairs)

    # --- what the floor allows, and which estimator the pooling step wants -------------
    raw = pd.read_csv(TOX21)
    raw = raw[~raw["luciferase_inhibitor"].astype(bool)]
    unfitted = raw[raw["pic50"].isna()]["activity_outcome"].value_counts().to_dict()
    print(
        f"\ntox21 CYP2D6 screened {len(raw):,}, fitted a pIC50 for {raw['pic50'].notna().sum():,} "
        f"({raw['pic50'].notna().mean():.0%}); the rest: {unfitted}"
    )

    pooled = tox[~tox["key"].isin(set(ch["key"]))].copy()
    offset = exact["delta"].mean()
    pooled["corrected"] = pooled["tox21"] + offset
    print(f"pooling would add {len(pooled):,} rows; {(pooled['tox21'] < 5.0).mean():.0%} of them sit below")
    print(f"    pIC50 5.0, where the anchors number {len(below)}.")
    print(
        f"    corrected: {(pooled['corrected'] < 4.0).sum():,} below 4.0, "
        f"{(pooled['corrected'] < 3.5).sum():,} below 3.5, min {pooled['corrected'].min():.2f}"
    )
    print(
        f"    scored column now: {(ch['challenge'] < 4.0).sum()} below 4.0, "
        f"{(ch['challenge'] < 3.5).sum()} below 3.5, min {ch['challenge'].min():.2f}"
    )

    slope, icept = np.polyfit(exact["tox21"], exact["challenge"], 1)
    print("\nE[challenge | tox21] is what a correction applied to a Tox21 reading needs.")
    print(f"    OLS slope {slope:.3f}, intercept {icept:.3f}; a pure shift is slope 1.")
    for x in (4.2, 4.6, 5.0, 6.0):
        print(
            f"    tox21 {x}: shift -> {x + offset:.2f}, fitted -> {slope * x + icept:.2f}, "
            f"gap {slope * x + icept - (x + offset):+.2f}"
        )


if __name__ == "__main__":
    main()
