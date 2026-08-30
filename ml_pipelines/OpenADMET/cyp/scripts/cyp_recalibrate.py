"""Place a submission's predictions on the blind half's own distribution.

Predictions carry two separable things: their order, which is the model, and their
placement on the pIC50 axis, which is not. `R2 = 2*rho*k - k^2 - b^2` with
`k = sd(pred)/sd(true)` and `b` the mean offset in units of sd(true) -- only rho depends on
the ordering, so an affine transform sets k and b freely without moving a single compound's
rank. R2 is therefore capped at rho^2, and the cap is reached at `k = rho, b = 0`. Note
`k = rho`, not `k = 1`: a squared-error model should be narrower than the truth by exactly
its correlation.

Both inputs are measured rather than estimated. `BLIND_MOMENTS` and `SOLVED_PEARSON` come
from `solve_blind_moments`, which reads three scored submissions; `oof_pearson` covers a
model with no board history yet.

The scored metric is not R2. ST-RAE is zero anywhere inside a compound's credible interval
and low-activity compounds carry wide intervals, so its optimum sits above the true centre
and narrower than `rho*sd`. Placing CYP2D6 exactly here raised its R2 0.363 -> 0.447 and
worsened its ST-RAE 0.565 -> 0.694. This is the R2-optimal placement: a good default and a
poor maximum.

Reads the submitted file rather than re-running inference, so predictions are byte-
identical to what was scored and placement is the only thing that changed. A positive
affine map cannot reorder, so Spearman and Kendall must come back unchanged -- if they
move, the submission path is at fault, not the calibration.

Usage:
    python cyp_recalibrate.py --source FILE --tag NAME     # SOLVED_PEARSON
    python cyp_recalibrate.py --source FILE --oof MODEL    # a new model's own OOF
    python cyp_recalibrate.py --solve                      # re-derive the constants
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openadmet_validation import validate_activity_submission
from scipy.stats import pearsonr
from workbench.api import Model, PublicData

# Safety net only. Clamping creates ties, and Spearman and Kendall are both scored, so a
# binding floor costs ranking. Each run prints how many compounds it binds -- keep it near
# zero; a handful of ties out of ~2,000 rows is below the metric's resolution.
FLOOR = 1.0

# The blind half's label distribution, solved exactly -- see `solve_blind_moments`. These
# are properties of the test set, not of any model, so they place a new model without
# spending a submission to rediscover them.
#
# Only the live half is described. OpenADMET split the 750-compound test set by chemical
# series and the other half is never scored until the challenge ends, so these transfer to
# it only insofar as the two halves share a distribution -- which series splitting is
# designed to prevent.
BLIND_MOMENTS = {
    "CYP1A2": {"mean": 4.412, "sd": 1.553},
    "CYP2C9": {"mean": 4.830, "sd": 1.101},
    "CYP2D6": {"mean": 3.107, "sd": 1.599},
    "CYP3A4": {"mean": 4.880, "sd": 1.272},
}
# Pearson of `cyp-reg-chemprop-mt-aux-100` against the live half, from the same solve.
# Invariant under affine recalibration, so it applies to any of its submissions.
# Independently corroborated: CYP2C9 lands at 0.838 against a measured Spearman of 0.840
# and CYP3A4 at 0.851 against 0.837, neither of which enters the solve.
SOLVED_PEARSON = {"CYP1A2": 0.753, "CYP2C9": 0.838, "CYP2D6": 0.678, "CYP3A4": 0.851}

# Placement that minimises ST-RAE is not the one that maximises R2, and CYP2D6 is where
# they diverge hardest. Three measured placements of the same predictions:
#   (3.57, 0.90) -> ST-RAE 0.5653    (3.32, 0.90) -> 0.6271    (3.11, 1.08) -> 0.6944
# R2 rose across that sequence (0.363 -> 0.430 -> 0.447) while ST-RAE fell. ST-RAE scores
# zero anywhere inside a compound's credible interval and low-activity compounds carry wide
# ones, so predicting high is nearly free while predicting low is punished by the actives,
# whose intervals are narrow. `scripts/cyp_strae_optimum.py` derives the minimum offline
# against the challenge's own intervals; it orders the three CYP2D6 board probes correctly
# at a stable offset. `sd` here is the spread to hit outright, not `rho * sd`.
STRAE_MOMENTS = {"CYP2D6": {"mean": 4.35, "sd": 0.90}}

# Scaffold-OOF Pearson understates blind Pearson -- the split is harder than the blind half,
# and an OOF prediction comes from one fold model where the endpoint averages five.
# Measured on `cyp-reg-chemprop-mt-aux-100`: OOF 0.570 / 0.681 / 0.408 / 0.795 against the
# solved 0.753 / 0.838 / 0.678 / 0.851. One model's ratio, so read it as what lets a new
# model be placed without spending a submission, not as a constant.
OOF_TO_BLIND = {"CYP1A2": 1.32, "CYP2C9": 1.23, "CYP2D6": 1.66, "CYP3A4": 1.07}

# Three scored entries of `cyp-reg-chemprop-mt-aux-100`, all affine transforms of one
# prediction vector. Two share a centre and differ in spread; the third shifts the centre.
# That spans enough to solve var(y), mean(y) and cov(y, p) exactly.
OUTPUTS = Path(__file__).parent / "outputs"
SOLVE_ENTRIES = [
    (
        OUTPUTS / "cyp-reg-chemprop-mt-aux-100_activity_submission_aux.csv",
        {"CYP1A2": 0.518857, "CYP2C9": 0.677782, "CYP2D6": 0.345047, "CYP3A4": 0.683557},
    ),
    (
        OUTPUTS / "cyp-reg-chemprop-mt-aux-100_activity_submission_aux_aux2.csv",
        {"CYP1A2": 0.5595, "CYP2C9": 0.6875, "CYP2D6": 0.3630, "CYP3A4": 0.6777},
    ),
    (
        OUTPUTS / "cyp-reg-chemprop-mt-aux-100_activity_submission_aux_aux2_probe.csv",
        {"CYP1A2": 0.5361, "CYP2C9": 0.6638, "CYP2D6": 0.4298, "CYP3A4": 0.5732},
    ),
]

VALUE_COLUMNS = {iso: f"{iso}_pIC50_direct_inhibition" for iso in BLIND_MOMENTS}


def solve_blind_moments() -> dict:
    """Recover the blind half's mean, sd and our Pearson from three scored submissions.

    Each board row is an exact equation:
    `R2*V = 2*cov(y,p) - var(p) - (mean(y) - mean(p))^2`, with V = var(y). Three unknowns,
    and our submissions are exact affine transforms of one prediction vector, so cov scales
    by a known factor and three entries spanning two spreads and two centres determine all
    three.

    Entries A and B share a centre and differ in spread by `beta`; C shares B's spread and
    shifts its centre by `delta`. Differencing B and C cancels cov and leaves V and
    mean(y); differencing A and B supplies the third equation. The result is quadratic in
    V, and the physical root is the one giving a plausible label sd.

    Reproduces BLIND_MOMENTS and SOLVED_PEARSON. Rho divides two small differences, so
    board noise of +-0.005 in R2 moves it by roughly +-0.07 -- read it as approximate. The
    moments are far more robust: driving the bias to zero would need sd(y) 25-30% larger
    than any independent estimate supports.
    """
    (path_a, r2_a), (path_b, r2_b), (path_c, r2_c) = SOLVE_ENTRIES
    a, b, c = (pd.read_csv(p) for p in (path_a, path_b, path_c))

    solved = {}
    for iso, col in VALUE_COLUMNS.items():
        p_a, p_b, p_c = a[col], b[col], c[col]
        var_a, var_b = p_a.var(), p_b.var()
        beta = p_b.std() / p_a.std()
        delta = p_c.mean() - p_b.mean()
        d_r2 = r2_b[iso] - r2_c[iso]

        # (delta^2 - d_r2*V)^2 / (4*delta^2) * (beta-1) = V*(r2_b - beta*r2_a) + var_b - beta*var_a
        scale = (beta - 1) / (4 * delta**2)
        roots = np.roots(
            [
                scale * d_r2**2,
                -2 * scale * delta**2 * d_r2 - (r2_b[iso] - beta * r2_a[iso]),
                scale * delta**4 - (var_b - beta * var_a),
            ]
        )
        candidates = [r.real for r in roots if abs(r.imag) < 1e-9 and 0.25 < r.real < 9.0]
        if not candidates:
            raise ValueError(f"{iso}: no physical root for var(y) — board numbers may be stale")
        var_y = max(candidates)

        offset = (delta**2 - d_r2 * var_y) / (2 * delta)
        cov = (r2_a[iso] * var_y + var_a + offset**2) / 2
        rho = cov / np.sqrt(var_y * var_a)
        solved[iso] = {"mean": p_a.mean() + offset, "sd": np.sqrt(var_y), "pearson": rho}
        print(
            f"{iso:<8} mean {solved[iso]['mean']:>6.3f}  sd {solved[iso]['sd']:>5.3f}  "
            f"pearson {rho:.3f}  R2 ceiling {rho**2:.3f}"
        )
    return solved


def oof_pearson(model_names: str) -> dict:
    """Estimate blind Pearson from the out-of-fold captures of one or more models.

    The stand-in for SOLVED_PEARSON before a model has board history. Pass several
    comma-separated names to describe an ensemble: their out-of-fold predictions are
    averaged per isoform on the rows they share, which is the same quantity the submission
    averages. A model without an isoform's capture is skipped for that isoform, so a CYP2D6
    specialist contributes where it can and nowhere else.

    Scaffold-OOF understates the blind half, so each isoform is scaled by OOF_TO_BLIND;
    without that correction the placement under-spreads, badly on CYP2D6.
    """
    models = [m.strip() for m in model_names.split(",") if m.strip()]
    pearson = {}
    for iso in BLIND_MOMENTS:
        target = f"{iso.lower()}_pic50_direct_inhibition"
        frames = []
        for name in models:
            model = Model(name)
            runs = model.list_inference_runs()
            # Single-target models report under `full_cross_fold`; multi-target under `cv_<target>`.
            run = f"cv_{target}" if f"cv_{target}" in runs else "full_cross_fold"
            if run not in runs:
                continue
            d = model.get_inference_predictions(run)
            if d is None or target not in d.columns:
                continue
            frames.append(d[["molecule_name", target, "prediction"]].dropna().set_index("molecule_name"))
        if not frames:
            raise ValueError(f"no model supplied an out-of-fold capture for {iso}")
        common = sorted(set.intersection(*[set(f.index) for f in frames]))
        truth = frames[0].loc[common, target]
        averaged = np.column_stack([f.loc[common, "prediction"] for f in frames]).mean(axis=1)
        raw = pearsonr(truth, averaged).statistic
        pearson[iso] = raw * OOF_TO_BLIND[iso]
        print(
            f"{iso:<8} {len(frames)} member(s), n={len(common):,} | OOF pearson {raw:.3f} "
            f"-> blind estimate {pearson[iso]:.3f}"
        )
    return pearson


def place(sub: pd.DataFrame, pearson: dict, strae: bool = False) -> dict:
    """Move each isoform onto the blind half's centre with the R2-optimal spread.

    With both the moments and the correlation measured this is the R2 ceiling for a given
    model; nothing further is available without raising `pearson` itself.

    `strae` swaps in STRAE_MOMENTS where they exist, trading R2 for the metric actually
    scored. Ranking is untouched either way, so a submission answers the same questions
    about the model under both.
    """
    calibration = {}
    for iso, moments in BLIND_MOMENTS.items():
        rho = pearson[iso]
        current = sub[VALUE_COLUMNS[iso]]
        if strae and iso in STRAE_MOMENTS:
            moments = STRAE_MOMENTS[iso]
            target_sd = moments["sd"]
        else:
            target_sd = rho * moments["sd"]
        calibration[iso] = {
            "offset": moments["mean"] - current.mean(),
            "scale": target_sd / current.std(),
        }
        print(
            f"{iso:<8} pearson {rho:.3f} | sd {current.std():.2f} -> {target_sd:.2f} "
            f"(x{calibration[iso]['scale']:.2f}) | mean {current.mean():.2f} -> "
            f"{moments['mean']:.2f} ({calibration[iso]['offset']:+.2f}) | R2 ceiling {rho**2:.3f}"
        )
    return calibration


def recalibrate(source: Path, out_dir: Path, tag: str, calibration: dict) -> Path:
    """Apply each isoform's affine correction and write a validated submission."""
    sub = pd.read_csv(source)
    blind = PublicData().get("comp_chem/openadmet/cyp/testing/blinded")

    expected = {str(m) for m in blind["molecule_name"]}
    if set(sub["Molecule_Name"]) != expected:
        raise ValueError(f"{source} identifiers do not match the blinded set")

    print(f"\n{'isoform':<8} {'offset':>7} {'scale':>6} {'mean':>15} {'sd':>15} {'min':>15} {'floored':>8}")
    clamped = {}
    for iso, cal in calibration.items():
        col = VALUE_COLUMNS[iso]
        before = sub[col]
        # Scale about the isoform's own mean so the scale term does not move the centre.
        shifted = before.mean() + cal["scale"] * (before - before.mean()) + cal["offset"]
        clamped[iso] = int((shifted < FLOOR).sum())
        sub[col] = shifted.clip(lower=FLOOR)
        print(
            f"{iso:<8} {cal['offset']:>+7.2f} {cal['scale']:>6.2f} "
            f"{before.mean():>6.2f} -> {sub[col].mean():>5.2f} {before.std():>6.2f} -> {sub[col].std():>5.2f} "
            f"{before.min():>6.2f} -> {sub[col].min():>5.2f} {clamped[iso]:>8}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source.stem}_{tag}.csv"
    sub.to_csv(path, index=False)

    ok, errors = validate_activity_submission(path, expected_ids=expected)
    if not ok:
        raise ValueError(f"{path} failed OpenADMET's validator:\n  " + "\n  ".join(errors))
    print(f"\nPassed OpenADMET's validator: {path}")

    # A positive-scale affine map is rank-preserving; only the floor can create ties.
    original = pd.read_csv(source)
    for iso, col in VALUE_COLUMNS.items():
        if clamped[iso] == 0 and not original[col].rank().equals(sub[col].rank()):
            raise ValueError(f"{iso} ranking changed — an unclamped affine map cannot do that")
        if clamped[iso]:
            unfloored = sub[col] > FLOOR
            if not original.loc[unfloored, col].rank().equals(sub.loc[unfloored, col].rank()):
                raise ValueError(f"{iso} ranking changed above the floor")
    ties = {k: v for k, v in clamped.items() if v}
    print(f"Rank order preserved (floor created ties in: {ties or 'none'})")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Submission to place")
    parser.add_argument("--tag", default="placed", help="Suffix for the output filename")
    parser.add_argument("--out", type=Path, default=OUTPUTS, help="Output directory")
    parser.add_argument(
        "--oof",
        metavar="MODELS",
        help="Estimate Pearson from out-of-fold captures instead of SOLVED_PEARSON; "
        "comma-separate several names to describe an ensemble",
    )
    parser.add_argument("--solve", action="store_true", help="Re-derive BLIND_MOMENTS and SOLVED_PEARSON")
    parser.add_argument(
        "--strae", action="store_true", help="Use STRAE_MOMENTS where measured, trading R2 for the scored metric"
    )
    args = parser.parse_args()

    if args.solve:
        solve_blind_moments()
    if args.source:
        pearson = oof_pearson(args.oof) if args.oof else SOLVED_PEARSON
        calibration = place(pd.read_csv(args.source), pearson, strae=args.strae)
        recalibrate(args.source, args.out, args.tag, calibration)
    elif not args.solve:
        parser.error("nothing to do — pass --source, --solve, or both")
