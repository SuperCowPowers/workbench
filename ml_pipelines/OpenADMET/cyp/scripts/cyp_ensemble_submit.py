"""Build a submission by combining several models' blind-set predictions.

Combining diverse models is the only thing that has repeatedly improved CYP2D6. Three
architectural hypotheses -- task weighting, cross-isoform representation sharing, descriptor
features -- each came back null on that isoform; this did not.

**Members are weighted per compound by calibrated confidence, not averaged flat.** Each
member's confidence is scaled by its own |conf-to-error correlation|, so a member whose
confidence has been shown to track its error gets more say on the rows where it is confident.
Out of fold that beats the flat mean on every isoform by more than the isoform resolves:

    isoform   flat mean   calibrated conf   delta    resolves at
    CYP1A2       0.6074            0.6569  +0.0495          0.043
    CYP2C9       0.7307            0.7832  +0.0525          0.031
    CYP2D6       0.4949            0.5583  +0.0634          0.056
    CYP3A4       0.8321            0.8574  +0.0253          0.018

Both the correlations and the fallback weights come from `EnsembleSimulator` at run time,
read off each member's out-of-fold capture, so a rebuilt member re-weights itself rather
than leaving a stale constant behind. Confidence is live on all 750 blinded compounds --
no member returns NaN there -- but `conf_weights_with_fallback` still guards the row,
since a compound the proximity backend cannot resolve would otherwise poison the average.

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

Predictions are combined, not the placements. Placement happens afterwards against the
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
from workbench.api import Endpoint, Model, PublicData
from workbench.api.inference_cache import InferenceCache
from workbench.utils.ensemble_simulator import EnsembleSimulator
from workbench.utils.ensemble_utils import conf_weights_with_fallback

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


MONROE_ENDPOINT = "smiles-to-monroe-v1"


def predict(model: str, blind: pd.DataFrame, embedded: pd.DataFrame) -> pd.DataFrame:
    """Blind-set predictions from one endpoint, indexed by compound.

    Chemprop featurizes from SMILES inside the endpoint. A model over a molecular embedding
    needs that embedding as a column, so it is handed the pre-embedded frame instead.
    """
    features = Model(model).features() or []
    source = embedded if "monroe" in features else blind
    out = Endpoint(model).inference(source[["molecule_name", "smiles"] + [f for f in features if f == "monroe"]].copy())
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


def confidence_for(preds: pd.DataFrame, iso: str) -> pd.Series:
    """The isoform's confidence column, named the same way `column_for` describes."""
    named = f"{iso.lower()}_pic50_direct_inhibition_confidence"
    if named in preds.columns:
        return preds[named]
    if "confidence" in preds.columns:
        return preds["confidence"]
    raise ValueError(f"no {iso} confidence column — found {list(preds.columns)[:8]}")


def member_scaling(iso: str, members: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-member `corr_scale` and fallback weights, read off the out-of-fold captures.

    Returns both arrays in `members` order. `corr_scale` is each member's
    |conf-to-error correlation|; the fallback weights are inverse-MAE, used on any row
    whose confidences carry no usable signal.
    """
    sim = EnsembleSimulator(members, id_column="molecule_name", target=f"{iso.lower()}_pic50_direct_inhibition")
    config = sim.get_best_strategy_config(select_by="spearman")
    if config["aggregation_strategy"] != "calibrated_conf_weighted":
        print(
            f"  WARNING {iso}: out-of-fold now prefers '{config['aggregation_strategy']}' "
            f"over calibrated_conf_weighted; this script still applies the latter"
        )
    return (
        np.array([config["corr_scale"][m] for m in members]),
        np.array([config["model_weights"][m] for m in members]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="ensemble", help="Output filename suffix")
    args = parser.parse_args()

    blind = PublicData().get("comp_chem/openadmet/cyp/testing/blinded")
    if len(blind) != N_TEST:
        raise ValueError(f"blinded set is {len(blind)} rows, expected {N_TEST}")

    # SMILES-keyed cache (S3-persisted), so this only embeds compounds it has not seen.
    embedded = InferenceCache(Endpoint(MONROE_ENDPOINT), auto_invalidate_cache=True).inference(
        blind[["molecule_name", "smiles"]].copy()
    )
    missing = embedded["monroe"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} of {N_TEST} blinded compounds have no Monroe embedding "
            f"({list(embedded.loc[missing, 'molecule_name'])[:5]}); every member must score every row"
        )

    every = sorted({m for members in MEMBERS.values() for m in members})
    print(f"Predicting {N_TEST} blinded compounds with {len(every)} models")
    preds = {m: predict(m, blind, embedded) for m in every}

    sub = pd.DataFrame({"SMILES": blind["smiles"].values, "Molecule_Name": blind["molecule_name"].values})
    print(f"\n{'isoform':<8}{'members':>9}{'mean':>8}{'sd':>7}{'spread':>8}{'fallback':>10}")
    for iso, members in MEMBERS.items():
        stack = np.column_stack([column_for(preds[m], iso).reindex(sub["Molecule_Name"]).to_numpy() for m in members])
        conf = np.column_stack(
            [confidence_for(preds[m], iso).reindex(sub["Molecule_Name"]).to_numpy() for m in members]
        )

        corr_scale, fallback_w = member_scaling(iso, members)
        weights = conf_weights_with_fallback(conf * corr_scale, fallback_w)
        sub[VALUE_COLUMNS[iso]] = (stack * weights).sum(axis=1)

        n_fallback = int((~np.isfinite(conf).all(axis=1)).sum())
        print(
            f"{iso:<8}{len(members):>9}{stack.mean():>8.2f}{sub[VALUE_COLUMNS[iso]].std():>7.2f}"
            f"{stack.std(axis=1).mean():>8.3f}{n_fallback:>10}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cyp-ensemble_activity_submission_{args.tag}.csv"
    sub.to_csv(path, index=False)
    ok, errors = validate_activity_submission(path, expected_ids={str(m) for m in blind["molecule_name"]})
    if not ok:
        raise ValueError(f"{path} failed OpenADMET's validator:\n  " + "\n  ".join(errors))
    print(f"\nPassed OpenADMET's validator: {path}")
