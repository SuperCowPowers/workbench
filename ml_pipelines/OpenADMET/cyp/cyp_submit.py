"""Build a Direct Inhibition submission file for the OpenADMET CYP challenge.

Predicts the 750 blinded compounds and writes the challenge's exact schema. Column
names here are the challenge's own (`CYP3A4_pIC50_direct_inhibition`, case-sensitive),
not the snake_cased names used everywhere else in this pipeline, so the mapping happens
in one place: MODEL_TO_SUBMISSION below.

The checks mirror OpenADMET's `validation/activity_validation.py` -- exact row count,
no missing or duplicate ids, finite floats -- because a rejected upload costs a 12-hour
submission window. Run their validator too before uploading; this is a first pass, not
a replacement.

SMILES are written back exactly as the challenge issued them rather than as our
standardized form, so the file round-trips their own identifiers.

Usage:
    python cyp_submit.py                          # default model, writes ./outputs/
    python cyp_submit.py --model NAME --out DIR
"""

import argparse
from pathlib import Path

import pandas as pd
from workbench.api import Endpoint, PublicData

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

# our column -> the challenge's column
MODEL_TO_SUBMISSION = {
    f"{iso.lower()}_pic50_direct_inhibition_pred": f"{iso}_pIC50_direct_inhibition" for iso in ISOFORMS
}
SUBMISSION_COLUMNS = ["SMILES", "Molecule_Name"] + list(MODEL_TO_SUBMISSION.values())
N_TEST = 750


def build_submission(model_name: str, out_dir: Path) -> Path:
    """Predict the blinded set and write a validated submission CSV."""
    blind = PublicData().get("comp_chem/openadmet/cyp/testing/blinded")
    if len(blind) != N_TEST:
        raise ValueError(f"Expected {N_TEST} blinded compounds, got {len(blind)}")

    preds = Endpoint(model_name).inference(blind[["molecule_name", "smiles"]])

    missing = [c for c in MODEL_TO_SUBMISSION if c not in preds.columns]
    if missing:
        raise ValueError(f"Endpoint '{model_name}' did not return {missing}. Is it the 4-target model?")

    sub = preds[["smiles", "molecule_name"] + list(MODEL_TO_SUBMISSION)].rename(
        columns={"smiles": "SMILES", "molecule_name": "Molecule_Name", **MODEL_TO_SUBMISSION}
    )
    # Their identifiers, not our standardized SMILES.
    sub["SMILES"] = sub["Molecule_Name"].map(dict(zip(blind["molecule_name"], blind["smiles"])))
    sub = sub[SUBMISSION_COLUMNS]

    validate(sub, set(blind["molecule_name"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_activity_submission.csv"
    sub.to_csv(path, index=False)
    return path


def validate(sub: pd.DataFrame, expected_ids: set) -> None:
    """Raise on anything OpenADMET's validator would reject.

    Args:
        sub (pd.DataFrame): The assembled submission frame
        expected_ids (set): The blinded set's Molecule_Name values

    Raises:
        ValueError: On row count, id, or value problems
    """
    problems = []
    if len(sub) != N_TEST:
        problems.append(f"{len(sub)} rows, expected {N_TEST}")
    if list(sub.columns) != SUBMISSION_COLUMNS:
        problems.append(f"columns are {list(sub.columns)}, expected {SUBMISSION_COLUMNS}")
    if sub["Molecule_Name"].duplicated().any():
        problems.append(f"{int(sub['Molecule_Name'].duplicated().sum())} duplicate Molecule_Name")

    submitted = set(sub["Molecule_Name"])
    if submitted != expected_ids:
        problems.append(f"id mismatch: {len(expected_ids - submitted)} missing, {len(submitted - expected_ids)} extra")

    for col in MODEL_TO_SUBMISSION.values():
        values = pd.to_numeric(sub[col], errors="coerce")
        if not values.notna().all():
            problems.append(f"{col}: {int(values.isna().sum())} non-numeric or missing")
        elif not values.between(-1e6, 1e6).all():
            problems.append(f"{col}: non-finite values")

    if problems:
        raise ValueError("Submission failed validation:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cyp-reg-chemprop-mt-100", help="Endpoint to predict with")
    parser.add_argument("--out", default="outputs", type=Path, help="Directory for the submission file")
    args = parser.parse_args()

    written = build_submission(args.model, args.out)
    frame = pd.read_csv(written)
    print(f"Wrote {written} — {len(frame)} rows")
    print(frame[list(MODEL_TO_SUBMISSION.values())].describe().loc[["mean", "min", "max"]].round(2).to_string())
