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

**The confidence correction is then extrapolated**, `(1 + lam) * base - lam * flat`, with
`lam` chosen per isoform on held-out hinge. `base - flat` is the correction itself, and out
of fold it is under-applied by a factor of 2.5 to 3.5. Because this combines two different
vectors rather than rescaling one, it reorders; an affine rescale could not move Spearman
at all. Out of fold, with each isoform's row-sampling bar:

    isoform   calibrated conf   + extrapolation   delta    resolves at   lam
    CYP1A2             0.6569            0.6937  +0.0368         0.033   2.0
    CYP2C9             0.7832            0.8118  +0.0286         0.028   1.5
    CYP2D6             0.5583            0.6200  +0.0617         0.037   2.5
    CYP3A4             0.8574            0.8722  +0.0148         0.012   1.5

All four clear their bar and `lam` holds across folds, so this is one effect rather than
four fits. Read the gains against how far each isoform transfers, though: CYP3A4 gains
least and transfers best, CYP2D6 gains most and transfers worst.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from cyp_ensemble import MEMBERS, combine, fit_lambda, member_scaling
from cyp_recalibrate import VALUE_COLUMNS
from openadmet_validation import validate_activity_submission
from workbench.api import Endpoint, Model, PublicData
from workbench.api.inference_cache import InferenceCache

OUT = Path(__file__).parent / "outputs"
N_TEST = 750

# Compounds per cache write. The cache persists between writes, so this bounds what an
# interrupted run loses -- roughly half an hour of the slowest member rather than hours.
SNAPSHOT = 64


MONROE_ENDPOINT = "smiles-to-monroe-v1"
READOUT_MODEL = "cyp-reg-chemprop-log2fc"
READOUTS = {f"{iso}_log2fc": f"{iso}_log2fc_readout" for iso in ("cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2")}


def add_readouts(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the predicted log2FC readouts the readout members take as features.

    `openadmet_cyp_readout_monroe_f1` gives screen compounds out-of-fold readouts and
    everything else the endpoint's. The blind 750 are disjoint from the screen, so they
    take endpoint predictions throughout, exactly as the Monroe embedding does.
    """
    preds = Endpoint(READOUT_MODEL).inference(frame[["molecule_name", "smiles"]].copy()).set_index("molecule_name")
    for target, feature in READOUTS.items():
        frame[feature] = frame["molecule_name"].map(preds[f"{target}_pred"])
    missing = frame[list(READOUTS.values())].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} of {len(frame)} compounds have no readout; every member scores every row"
        )
    return frame


def predict(model: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Blind-set predictions from one endpoint, indexed by compound.

    Chemprop featurizes from SMILES inside the endpoint; a model over the Monroe embedding
    or the screen readouts needs those as columns. Each endpoint is handed exactly the
    features it declares, so a member whose features the frame does not carry raises here
    rather than inside the endpoint.

    Cached per SMILES, which is sound because every feature a member reads is itself a
    deterministic function of the SMILES. An Atom-JEPA member runs tens of seconds per
    compound on its CPU pool, so a run that dies partway through leaves its finished
    compounds banked and the next one picks up where it stopped.
    """
    features = [f for f in (Model(model).features() or []) if f != "smiles"]
    absent = [f for f in features if f not in frame.columns]
    if absent:
        raise ValueError(f"{model} declares features the blind frame does not carry: {absent}")
    cached = InferenceCache(Endpoint(model), auto_invalidate_cache=True, snapshot=SNAPSHOT)
    out = cached.inference(frame[["molecule_name", "smiles"] + features].copy())
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="ensemble", help="Output filename suffix")
    parser.add_argument("--flat", action="store_true", help="Unweighted member mean, ignoring confidence and lambda")
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

    embedded = add_readouts(embedded)

    # Members score concurrently. Each one is a separate endpoint with its own fleet, so
    # they contend for nothing, and the work here is endpoint invocation and S3 polling
    # rather than local compute -- the Atom-JEPA members run tens of seconds per compound,
    # so scoring them one after another would spend most of a day waiting on idle fleets.
    # Every member is attempted before anything raises, which leaves each one's finished
    # compounds in its cache for the next run.
    every = sorted({m for members in MEMBERS.values() for m in members})
    print(f"Predicting {N_TEST} blinded compounds with {len(every)} models")
    preds, failed = {}, {}
    with ThreadPoolExecutor(max_workers=len(every)) as pool:
        pending = {pool.submit(predict, m, embedded): m for m in every}
        for future in as_completed(pending):
            name = pending[future]
            try:
                preds[name] = future.result()
                print(f"  {name}: done ({len(preds)}/{len(every)})")
            except Exception as exc:
                failed[name] = exc
                print(f"  {name}: FAILED {type(exc).__name__}: {exc}")
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(every)} members did not score every compound "
            f"({', '.join(sorted(failed))}); re-run to resume from their caches"
        )

    sub = pd.DataFrame({"SMILES": blind["smiles"].values, "Molecule_Name": blind["molecule_name"].values})
    print(f"\n{'isoform':<8}{'lambda':>9}{'mean':>8}{'sd':>7}{'spread':>8}{'fallback':>10}")
    for iso, members in MEMBERS.items():
        stack = np.column_stack([column_for(preds[m], iso).reindex(sub["Molecule_Name"]).to_numpy() for m in members])
        conf = np.column_stack(
            [confidence_for(preds[m], iso).reindex(sub["Molecule_Name"]).to_numpy() for m in members]
        )

        if args.flat:
            corr_scale = fallback_w = np.ones(len(members))
            lam = 0.0
        else:
            corr_scale, fallback_w = member_scaling(iso, members)
            lam = fit_lambda(iso, members, corr_scale, fallback_w)
        sub[VALUE_COLUMNS[iso]] = combine(stack, conf, corr_scale, fallback_w, lam, flat=args.flat)

        n_fallback = int((~np.isfinite(conf).all(axis=1)).sum())
        print(
            f"{iso:<8}{lam:>9.2f}{stack.mean():>8.2f}{sub[VALUE_COLUMNS[iso]].std():>7.2f}"
            f"{stack.std(axis=1).mean():>8.3f}{n_fallback:>10}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cyp-ensemble_activity_submission_{args.tag}.csv"
    sub.to_csv(path, index=False)
    ok, errors = validate_activity_submission(path, expected_ids={str(m) for m in blind["molecule_name"]})
    if not ok:
        raise ValueError(f"{path} failed OpenADMET's validator:\n  " + "\n  ".join(errors))
    print(f"\nPassed OpenADMET's validator: {path}")
