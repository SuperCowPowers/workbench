"""Pull the Tox21 CYP inhibition screen into public-data CSVs.

A 15-point qHTS dose-response screen of the Tox21 10K library against five human CYP
isoforms (1A2, 2C9, 2C19, 2D6, 3A4), run in triplicate on P450-Glo bioluminescent
substrates. One assay ID per isoform rather than one panel, so the isoform comes from the
AID rather than from a column.

Complements the Veith panel (`pull_pubchem_cyp_data.py`, AID 1851) rather than duplicating
it: different library, different detection chemistry, and roughly 5.6k compounds Veith and
ChEMBL between them do not cover. It is also the only public CYP source we have found that
is weak-inhibitor-rich -- the median fitted pIC50 is 4.76 and 63% of actives fall below
5.0, where ChEMBL bottoms out at 4.0 and the OpenADMET challenge set is hit-enriched.

Bioluminescent CYP assays report firefly-luciferase inhibitors as CYP inhibitors, so the
matching counter-screen (AID 1224835) rides along as `luciferase_inhibitor`. Filter on it
before modeling; it is carried rather than applied so the choice stays with the consumer.

Triplicate is kept as an uncertainty estimate rather than collapsed silently: `pic50` is
the median over replicates that fitted, with `pic50_n` and `pic50_sd` alongside.

Writes `output/comp_chem/tox21/cyp_inhibition/` -- one long file over all isoforms plus a
per-isoform file -- and the same set again under `censored/`, where the rows the screen calls
inactive carry a bound instead of a null. Those rows are already present in both families; only
the label differs. `pic50` holds the pIC50 implied by the highest concentration that row was
read at and `pic50_lt` marks it, so the true value is at or below the label. This screen runs
to a higher top concentration than the Veith panel, so its bounds reach further down. Train
these with `bounded_loss=True`; with bounded loss off a bound reads as an exact measurement.
The shared `upload_data.py` then publishes both and merges the matching `descriptions.json`
entries.

Run:
    python pull_tox21_cyp_data.py
    AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply   # then publish
"""

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pull_common import censoring_bound, standardize_smiles, top_tested_concentration

log = logging.getLogger("workbench")

OUTPUT_DIR = Path(__file__).parent / "output" / "comp_chem" / "tox21" / "cyp_inhibition"

REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# One assay per isoform. Titles confirm the mapping; it is not derivable from the numbers.
AIDS = {
    "cyp1a2": 1671199,
    "cyp2c9": 1671198,
    "cyp2c19": 1671197,
    "cyp2d6": 1671196,
    "cyp3a4": 1671201,
}
LUCIFERASE_AID = 1224835

SUMMARY = {
    "PUBCHEM_SID": "sid",
    "PUBCHEM_CID": "cid",
    "PUBCHEM_EXT_DATASOURCE_SMILES": "orig_smiles",
    "PUBCHEM_ACTIVITY_OUTCOME": "activity_outcome",
}
# Per-replicate readouts worth keeping. The per-concentration "Activity at ... uM" columns
# are dropped -- 15 points x 3 replicates x 5 assays, and the fitted curve already
# summarizes them.
REPLICATE = ["Potency", "Efficacy", "Fit_R2", "Fit_HillSlope", "Fit_CurveClass", "Max_Response"]


def fetch_assay(aid: int) -> pd.DataFrame:
    """One assay's full result table, minus PubChem's type/description preamble rows."""
    resp = requests.get(f"{REST}/assay/aid/{aid}/CSV", timeout=900)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    return raw[pd.to_numeric(raw["PUBCHEM_SID"], errors="coerce").notna()].copy()


def across_replicates(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """The replicate columns for one readout, as a numeric frame."""
    cols = [c for c in raw.columns if c.startswith(f"{field}-Replicate_")]
    return raw[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=raw.index)


def collapse(raw: pd.DataFrame) -> pd.DataFrame:
    """Summary columns plus each replicated readout reduced to its median."""
    df = raw[list(SUMMARY)].rename(columns=SUMMARY)
    for field in REPLICATE:
        df[field.lower()] = across_replicates(raw, field).median(axis=1)

    # pIC50 comes from Potency, not Fit_LogAC50. Both encode the same AC50 -- and agree
    # where both exist -- but Fit_LogAC50 is written for every curve the fitter touched,
    # including compounds the assay calls inactive, where it is an extrapolation past the
    # top concentration. Potency is reported only where the curve class supports it.
    potency = across_replicates(raw, "Potency")
    pic50 = 6 - np.log10(potency)
    df["pic50"] = pic50.median(axis=1)
    df["pic50_n"] = potency.notna().sum(axis=1)
    df["pic50_sd"] = pic50.std(axis=1)

    # Fit_LogAC50 is log10(AC50) in molar and must equal -pIC50 replicate for replicate.
    # Compared per replicate rather than on the medians, which are taken over different
    # subsets: a compound can fit three curves but earn only one Potency.
    logac50 = across_replicates(raw, "Fit_LogAC50")
    for i, col in enumerate(pic50.columns):
        if i >= logac50.shape[1]:
            break
        pair = pd.DataFrame({"a": pic50[col], "b": -logac50.iloc[:, i]}).dropna()
        if not len(pair):
            continue
        # Fraction rather than max: agreement is exact to 4 decimals for all but a handful
        # of sub-nanomolar rows, where Potency's significant figures run out. A unit change
        # or a column swap moves every row by log units, so it trips this immediately.
        off = float(((pair["a"] - pair["b"]).abs() > 0.05).mean())
        if off > 0.001:
            raise ValueError(f"replicate {i + 1}: {off:.1%} of rows disagree with fit_logac50 — units moved")

    phenotype = [c for c in raw.columns if c.startswith("Phenotype-Replicate_")]
    df["phenotype"] = raw[phenotype].mode(axis=1)[0] if phenotype else pd.NA

    # Potency is reported per replicate, so a compound whose summary call is Inactive can
    # still carry a number from the one replicate that fitted -- 1,221 of them on CYP2D6,
    # 280 of those phenotyped Inhibitor. Publishing those as potency would invent values
    # the assay declined to stand behind, so the potency fields follow the summary call.
    # `pic50_n` and `pic50_sd` are kept for every row, so the raw disagreement stays visible.
    inactive = df["activity_outcome"] != "Active"
    df.loc[inactive, ["potency", "pic50"]] = np.nan

    # Read off `raw` before the per-concentration columns go: an inactive call is the
    # observation that the AC50 lies above the point the assay stopped at.
    df["censor_bound"] = censoring_bound(top_tested_concentration(raw))
    return df


def luciferase_inhibitors() -> set:
    """SIDs the firefly-luciferase counter-screen calls active."""
    raw = fetch_assay(LUCIFERASE_AID)
    active = raw[raw["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"]["PUBCHEM_SID"]
    sids = set(pd.to_numeric(active, errors="coerce").dropna().astype("Int64"))
    log.info(f"Luciferase counter-screen (AID {LUCIFERASE_AID}): {len(sids):,} inhibitors")
    return sids


def pull_tox21_cyp() -> dict[str, pd.DataFrame]:
    """Fetch every isoform assay, then split into a long table plus one table per isoform."""
    flagged = luciferase_inhibitors()
    frames = []
    for isoform, aid in AIDS.items():
        log.info(f"Fetching {isoform} (AID {aid})")
        df = collapse(fetch_assay(aid))
        df.insert(0, "isoform", isoform)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    for col in ["sid", "cid"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["luciferase_inhibitor"] = df["sid"].isin(flagged)

    structures = df[["orig_smiles"]].dropna().drop_duplicates()
    structures["smiles"] = structures["orig_smiles"].apply(standardize_smiles)
    df = df.merge(structures, on="orig_smiles", how="left")
    dropped = int(df["smiles"].isna().sum())
    if dropped:
        log.warning(f"Dropping {dropped:,} rows whose SMILES could not be standardized")
        df = df[df["smiles"].notna()]
    df.insert(3, "smiles", df.pop("smiles"))
    df = df.sort_values(["sid", "isoform"]).reset_index(drop=True)

    return {"": split_by_isoform(df.drop(columns=["censor_bound"])), "censored": split_by_isoform(apply_censoring(df))}


def apply_censoring(df: pd.DataFrame) -> pd.DataFrame:
    """Give the inactive rows the bound the assay stopped at, and flag them.

    A compound the screen calls inactive was read to the top of the series without a
    dose-response, so its true pIC50 is at or below what that concentration implies.
    Inconclusive rows are left null -- the assay did not decide, so there is no bound.
    """
    out = df.copy()
    censored = out["activity_outcome"].eq("Inactive") & out["pic50"].isna() & out["censor_bound"].notna()
    out["pic50"] = out["pic50"].where(~censored, out["censor_bound"])
    out.insert(out.columns.get_loc("pic50") + 1, "pic50_lt", censored)
    return out.drop(columns=["censor_bound"])


def split_by_isoform(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The long table plus one table per isoform."""
    out = {"all_isoforms": df}
    for isoform in AIDS:
        out[isoform] = df[df["isoform"] == isoform].drop(columns=["isoform"]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Pull the Tox21 CYP inhibition screen from PubChem")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    skeleton = {}
    print("\n" + "=" * 70)
    print(f"Tox21 CYP Inhibition Pull (AIDs {', '.join(str(a) for a in AIDS.values())})")
    print("=" * 70)
    for subdir, family in pull_tox21_cyp().items():
        for name, df in family.items():
            out_path = args.output_dir / subdir / f"{name}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            labelled = int(df["pic50"].notna().sum())
            bounds = int(df["pic50_lt"].sum()) if "pic50_lt" in df.columns else 0
            log.info(
                f"  {subdir + '/' if subdir else ''}{name}.csv  ({len(df):,} rows, "
                f"{labelled:,} labelled, {bounds:,} of them bounds)"
            )
            entry = {
                "num_compounds": int(df["sid"].nunique()),
                "license": "public-domain",
                "columns": {c: "" for c in df.columns},
            }
            if bounds:
                entry["censoring"] = {
                    "pic50": {
                        "direction": "left",
                        "flag_column": "pic50_lt",
                        "n_censored": bounds,
                        "bound_source": "highest concentration tested, per record",
                    }
                }
            key = "/".join(part for part in ("comp_chem/tox21/cyp_inhibition", subdir, f"{name}.csv") if part)
            skeleton[key] = entry

    print(f"\nWrote {len(skeleton)} files -> {args.output_dir}")
    print("\ndescriptions.json skeleton (fill in column meanings, merge into descriptions.json):")
    print(json.dumps(skeleton, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
