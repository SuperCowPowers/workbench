"""Build a submission by averaging several models' blind-set predictions.

Averaging diverse models is the only thing that has repeatedly improved CYP2D6. Three
architectural hypotheses -- task weighting, cross-isoform representation sharing, descriptor
features -- each came back null on that isoform; this did not.

Membership is per isoform because the CYP2D6 specialists have no other heads. Members are
chosen by architecture rather than by score, since picking the best-scoring subset out of many
overfits the ruler used to pick it. What earns a slot is decorrelation: the two multi-isoform
chemprops agree with each other at rho 0.93-0.97, while TabICL over the Monroe embedding sits
at 0.73-0.89 against them.

Out of fold, against the two chemprops alone (four for CYP2D6):

    isoform   chemprops   + tox + tabicl     AUC <4.0 vs >=4.5
    CYP1A2       0.5864            0.6154     0.858 -> 0.876
    CYP2C9       0.7115            0.7342     0.925 -> 0.934
    CYP2D6       0.4975            0.5137     0.797 -> 0.815
    CYP3A4       0.8255            0.8398     0.949 -> 0.953

No single delta clears its isoform's resolution threshold (0.043 / 0.031 / 0.056 / 0.018), so
read the pattern rather than any row: both additions are positive on all four isoforms.

Predictions are averaged, not the placements. Placement happens afterwards against the
ensemble's own out-of-fold correlation:

    python cyp_ensemble_submit.py
    python cyp_recalibrate.py --source outputs/<written file> --oof <the same members> --strae
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from cyp_recalibrate import VALUE_COLUMNS
from openadmet_validation import validate_activity_submission
from workbench.api import Endpoint, PublicData

OUT = Path(__file__).parent / "outputs"
N_TEST = 750

MT = "cyp-reg-chemprop-union-p30"
AUX = "cyp-reg-chemprop-mt-aux-100"
TOX = "cyp-reg-chemprop-union-p30-tox"


def tabicl(iso: str) -> str:
    """TabICL over the Monroe embedding, one model per isoform."""
    return f"cyp-reg-tabicl-{iso.lower().removeprefix('cyp')}-monroe"


MEMBERS = {
    "CYP1A2": [MT, AUX, TOX, tabicl("CYP1A2")],
    "CYP2C9": [MT, AUX, TOX, tabicl("CYP2C9")],
    "CYP2D6": [MT, AUX, TOX, tabicl("CYP2D6"), "cyp-reg-chemprop-2d6-isoform", "cyp-reg-chemprop-2d6-single"],
    "CYP3A4": [MT, AUX, TOX, tabicl("CYP3A4")],
}


def predict(model: str, blind: pd.DataFrame) -> pd.DataFrame:
    """Blind-set predictions from one endpoint, indexed by compound."""
    out = Endpoint(model).inference(blind[["molecule_name", "smiles"]].copy())
    return out.set_index("molecule_name")


def column_for(preds: pd.DataFrame, iso: str) -> pd.Series:
    """The isoform's prediction column, whatever the model calls it.

    Multi-target models suffix each target with `_pred`; a single-target model writes a
    bare `prediction`.
    """
    named = f"{iso.lower()}_pic50_direct_inhibition_pred"
    if named in preds.columns:
        return preds[named]
    if "prediction" in preds.columns:
        return preds["prediction"]
    raise ValueError(f"no {iso} prediction column — found {list(preds.columns)[:8]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="ensemble", help="Output filename suffix")
    args = parser.parse_args()

    blind = PublicData().get("comp_chem/openadmet/cyp/testing/blinded")
    if len(blind) != N_TEST:
        raise ValueError(f"blinded set is {len(blind)} rows, expected {N_TEST}")

    every = sorted({m for members in MEMBERS.values() for m in members})
    print(f"Predicting {N_TEST} blinded compounds with {len(every)} models")
    preds = {m: predict(m, blind) for m in every}

    sub = pd.DataFrame({"SMILES": blind["smiles"].values, "Molecule_Name": blind["molecule_name"].values})
    print(f"\n{'isoform':<8}{'members':>9}{'mean':>8}{'sd':>7}   spread across members")
    for iso, members in MEMBERS.items():
        cols = [column_for(preds[m], iso).reindex(sub["Molecule_Name"]).to_numpy() for m in members]
        stack = np.column_stack(cols)
        sub[VALUE_COLUMNS[iso]] = stack.mean(axis=1)
        disagreement = stack.std(axis=1).mean()
        print(f"{iso:<8}{len(members):>9}{stack.mean():>8.2f}{stack.mean(axis=1).std():>7.2f}   {disagreement:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cyp-ensemble_activity_submission_{args.tag}.csv"
    sub.to_csv(path, index=False)
    ok, errors = validate_activity_submission(path, expected_ids={str(m) for m in blind["molecule_name"]})
    if not ok:
        raise ValueError(f"{path} failed OpenADMET's validator:\n  " + "\n  ".join(errors))
    print(f"\nPassed OpenADMET's validator: {path}")
