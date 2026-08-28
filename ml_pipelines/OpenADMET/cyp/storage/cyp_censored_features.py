"""Producer: the CYP regression FeatureSet with censored CYP2D6 labels.

Training on this did not beat the uncensored baseline -- `cyp_chemprop_mt_censored.py`
records what happened and why the single-concentration readout cannot help CYP2D6.
Kept because the construction is sound and the negative result depends on it.

`openadmet_cyp_f1` carries only the 1,493 CYP2D6 compounds that produced a fitted
dose-response curve. The single-concentration arm measured all 4,375 compounds against
CYP2D6 at 50 uM, and the ~2,880 that never inhibited are real measurements that the
regression track drops on the floor. They are the low end of the activity range, and a
model trained without them centers CYP2D6 predictions on the fitted-label mean (4.78)
with a floor at 4.15 -- against a blind set whose CYP2D6 labels sit far lower.

Those rows come back as left-censored labels: the target column holds `BOUND` and
`{target}_lt` marks the row, so chemprop's bounded loss penalises a prediction only
when it rises above the bound. Set `bounded_loss=True` on the model to honor them;
without it the columns are ignored and this FeatureSet trains identically to
`openadmet_cyp_f1`, which is what keeps the A/B clean.

BOUND sits above the assay's own detection limit (50 uM -> pIC50 4.31). The
single-concentration readout separates CYP2D6 actives from inactives poorly, so a bound
at the detection limit is only 34% pure and would punish the model for correctly calling
genuine weak inhibitors. Purity against grip is the whole trade: 4.31 is 34% pure and
constrains 98% of our current predictions, 5.0 is 89% pure and constrains 20%. 4.6 is
the knee at 78% pure over 62% -- past it purity gains flatten while grip falls away.
Every no-fit row clears the cut at any of these bounds, so the row count is not part of
the trade.

Only CYP2D6 is censored here. The other three isoforms rank 7, 11 and 22 on the
leaderboard against CYP2D6's 39, and a single-target change keeps the analog-holdout
comparison interpretable.

Rows in the analog holdout keep their original labels. A censored label there would be
scored as though it were exact, since the holdout capture reads the target column
directly.

Run after cyp_feature_sets.py:  python cyp_censored_features.py
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from workbench.api import DataSource, FeatureSet, PublicData
from workbench.training.splits import analog_holdout_split

SOURCE_FS = "openadmet_cyp_f1"
FS_NAME = "openadmet_cyp_censored_f1"

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]

CENSORED_ISOFORM = "CYP2D6"
CENSORED_TARGET = "cyp2d6_pic50_direct_inhibition"
BOUND = 4.6

df = FeatureSet(SOURCE_FS).pull_dataframe()
single_conc = PublicData().get("comp_chem/openadmet/cyp/training/single_concentration")
sc = single_conc[single_conc["enzyme"] == CENSORED_ISOFORM][["molecule_name", "log2fc_estimate"]]

# The log2fc -> pIC50 scale differs per isoform (assay window, substrate, background), so
# the "no inhibition" cut is fit from the compounds that have both readings rather than
# assumed from the -1 = half-signal identity.
paired = sc.merge(df[["molecule_name", CENSORED_TARGET]].dropna(), on="molecule_name")
iso_fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
    -paired["log2fc_estimate"].to_numpy(), paired[CENSORED_TARGET].to_numpy()
)
grid = np.linspace(-6.0, 1.0, 4001)
cut = float(grid[np.argmin(np.abs(iso_fit.predict(-grid) - BOUND))])

# Purity of the rule on the compounds we can check it against.
called_inactive = paired[paired["log2fc_estimate"] > cut][CENSORED_TARGET]
purity = 100.0 * (called_inactive <= BOUND).mean() if len(called_inactive) else float("nan")
print(f"log2fc cut for pIC50 <= {BOUND}: {cut:.2f}  ({purity:.1f}% of fitted compounds above it are truly <= {BOUND})")

# The holdout is derived from target values and compound identity, neither of which this
# script changes — recomputing it here reproduces the split the models are scored on.
holdout_names = set(
    df[analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)]["molecule_name"]
)

log2fc = df["molecule_name"].map(dict(zip(sc["molecule_name"], sc["log2fc_estimate"])))
censor = df[CENSORED_TARGET].isna() & log2fc.notna() & (log2fc > cut) & ~df["molecule_name"].isin(holdout_names)

out = df.copy()
out[CENSORED_TARGET] = out[CENSORED_TARGET].where(~censor, BOUND)
out[f"{CENSORED_TARGET}_lt"] = censor

fitted = int(df[CENSORED_TARGET].notna().sum())
print(f"CYP2D6: {fitted:,} fitted + {int(censor.sum()):,} censored = {fitted + int(censor.sum()):,} labelled rows")
in_holdout = df["molecule_name"].isin(holdout_names)
held = int((df[CENSORED_TARGET].isna() & log2fc.notna() & (log2fc > cut) & in_holdout).sum())
print(f"  {held} censorable rows left alone because they sit in the analog holdout")
labels = out.loc[out[CENSORED_TARGET].notna(), CENSORED_TARGET]
frac = 100 * (labels <= BOUND).mean()
print(f"  labels: mean {labels.mean():.2f}  sd {labels.std():.2f}  frac<={BOUND} {frac:.1f}%")

DataSource(out, name=f"{FS_NAME}_ds").to_features(
    FS_NAME, id_column="molecule_name", tags=["openadmet_cyp", "multi_task", "activity", "censored"]
)
print(f"Built '{FS_NAME}': {len(out):,} rows")
