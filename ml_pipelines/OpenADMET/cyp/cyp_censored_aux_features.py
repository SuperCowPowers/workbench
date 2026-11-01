"""Producer: the aux CYP FeatureSet plus censored pIC50 bounds on three isoforms.

The single-concentration arm measured every compound at 50 uM, so a compound with no fitted
dose-response curve still has evidence it is weak. Where that evidence is reliable, the pIC50
target takes the bound and `{target}_lt` marks the row, for `bounded_loss=True` to penalise a
prediction only when it crosses the bound.

The log2fc aux targets stay. They supervise the shared encoder on the scale the assay actually
measured; a bound does something they cannot, which is give a pIC50 head a label on the ~1,200
to 2,500 rows per isoform that have no fitted curve at all.

Reliability differs sharply by isoform. Of the paired compounds each cut calls inactive, the
fraction truly at or below the bound is 97% for CYP1A2, 96% for CYP3A4 and 91% for CYP2C9, but
34% for CYP2D6 -- its readout saturates at the inactive end, so a cut there is close to a coin
flip. CYP2D6 keeps fitted curves only.

Each bound sits below its isoform's blind-set mean, since a bound above the mean constrains
almost nothing. The log2fc cut for a bound is read off an isotonic fit of -log2fc against
fitted pIC50 rather than assumed, because the log2fc scale differs per isoform. Purity is
recomputed and printed on every build, so a cut that stops holding is visible rather than
silent.

Rows in the analog holdout keep their original labels -- the holdout capture reads the target
column directly and would score a bound as though it were exact.

Run after cyp_aux_features.py:  python cyp_censored_aux_features.py
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from workbench.api import DataSource, FeatureSet, PublicData
from workbench.training.splits import analog_holdout_split

SOURCE_FS = "openadmet_cyp_aux_f1"
FS_NAME = "openadmet_cyp_censored_f1"

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]

# pIC50 bound per censored isoform, keyed by the enzyme name the single-concentration arm uses.
# Each sits below that isoform's blind-set mean (1A2 4.41, 2C9 4.83, 3A4 4.88). CYP2D6 is absent
# on purpose; see the module docstring.
BOUNDS = {"CYP1A2": 4.31, "CYP2C9": 4.6, "CYP3A4": 4.6}

# Refuse to ship a cut whose calls are mostly wrong.
MIN_PURITY = 85.0

CUT_GRID = np.linspace(-6.0, 1.0, 4001)

df = FeatureSet(SOURCE_FS).pull_dataframe()
single_conc = PublicData().get("comp_chem/openadmet/cyp/training/single_concentration")

holdout = set(df[analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)]["molecule_name"])
in_holdout = df["molecule_name"].isin(holdout)

out = df.copy()
print(f"{'isoform':<9}{'bound':>7}{'cut':>8}{'checkable':>11}{'purity':>9}{'fitted':>9}{'censored':>10}{'held':>7}")
for enzyme, bound in BOUNDS.items():
    iso = enzyme.lower()
    target = f"{iso}_pic50_direct_inhibition"
    sc = single_conc[single_conc["enzyme"] == enzyme][["molecule_name", "log2fc_estimate"]]

    # The log2fc -> pIC50 scale differs per isoform (assay window, substrate, background), so the
    # cut is fit from the compounds carrying both readings.
    paired = sc.merge(df[["molecule_name", target]].dropna(), on="molecule_name")
    fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        -paired["log2fc_estimate"].to_numpy(), paired[target].to_numpy()
    )
    cut = float(CUT_GRID[np.argmin(np.abs(fit.predict(-CUT_GRID) - bound))])

    called = paired.loc[paired["log2fc_estimate"] > cut, target]
    purity = 100.0 * (called <= bound).mean() if len(called) else float("nan")
    if not purity >= MIN_PURITY:
        raise ValueError(
            f"{enzyme}: cut {cut:.2f} for bound {bound} is only {purity:.1f}% pure over "
            f"{len(called)} checkable compounds, under the {MIN_PURITY}% floor"
        )

    log2fc = df["molecule_name"].map(dict(zip(sc["molecule_name"], sc["log2fc_estimate"])))
    censor = df[target].isna() & log2fc.notna() & (log2fc > cut) & ~in_holdout

    fitted = int(df[target].notna().sum())
    out[target] = out[target].where(~censor, bound)
    out[f"{target}_lt"] = censor
    held = int((df[target].isna() & log2fc.notna() & (log2fc > cut) & in_holdout).sum())
    print(
        f"{enzyme:<9}{bound:>7}{cut:>8.2f}{len(called):>11,}{purity:>8.1f}%{fitted:>9,}"
        f"{int(censor.sum()):>10,}{held:>7}"
    )

# CYP2D6 carries no bounds, but chemprop reads one flag column per target, so it needs the column.
out["cyp2d6_pic50_direct_inhibition_lt"] = False

for target in TARGETS:
    labelled = out[target].notna()
    print(
        f"{target:<34} {int(labelled.sum()):>6,} labelled "
        f"({int(out[f'{target}_lt'].sum()):>5,} censored)  "
        f"mean {out.loc[labelled, target].mean():.2f}  sd {out.loc[labelled, target].std():.2f}"
    )

DataSource(out, name=f"{FS_NAME}_ds").to_features(
    FS_NAME, id_column="molecule_name", tags=["openadmet_cyp", "multi_task", "activity", "censored"]
)
print(f"Built '{FS_NAME}': {len(out):,} rows")
