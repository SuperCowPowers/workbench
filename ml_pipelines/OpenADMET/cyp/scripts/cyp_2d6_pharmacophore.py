"""Is CYP2D6 error concentrated in the chemistry the isoform is known to select for?

CYP2D6's active site binds a protonated basic nitrogen against Asp301/Glu216, so its
inhibitors are dominated by basic amines with an aromatic ring a few bonds away. CYP3A4
and CYP2C9 have no such requirement. If that pharmacophore carries the signal and a
SMILES-only encoder infers it poorly, error should concentrate in basic amines on CYP2D6
and nowhere else -- which would explain why cross-isoform sharing, extra rows and every
architecture change came back null there.

Splits out-of-fold error by basic-nitrogen content on all four isoforms, so CYP2D6 is read
against three controls rather than on its own. A gap that shows up everywhere is a property
of the models, not of the isoform.

Also reports where the credible intervals sit relative to our predictions, which is what
decides whether CYP2D6's 2.3% CI hit rate below pIC50 4.0 is us predicting high or the
intervals being unusual.

    python cyp_2d6_pharmacophore.py
"""

import numpy as np
from cyp_ensemble_submit import MEMBERS
from cyp_error_decomposition import CI_SOURCE, oof_average
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr
from workbench.api import FeatureSet

RDLogger.DisableLog("rdApp.*")
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

# Aliphatic amines that carry a positive charge at physiological pH. Amides, anilines and
# aromatic N are excluded: they are not basic enough to make the Asp301 salt bridge.
BASIC_N = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N=*);!$(N#*)]")


def basic_amine(smiles: str) -> bool:
    """True when the molecule carries an aliphatic amine, protonated at pH 7.4."""
    mol = Chem.MolFromSmiles(smiles)
    return bool(mol and mol.HasSubstructMatch(BASIC_N))


fs = FeatureSet(CI_SOURCE).pull_dataframe().set_index("molecule_name")
fs["basic"] = [basic_amine(s) for s in fs["smiles"]]
fs["mw"] = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in fs["smiles"]]

print(f"{'isoform':<9}{'group':<14}{'n':>6}{'spearman':>11}{'pearson':>9}{'MAE':>8}{'bias':>8}{'MW':>7}")
for iso in ISOFORMS:
    target = f"{iso.lower()}_pic50_direct_inhibition"
    df = oof_average(MEMBERS[iso], target)
    for col in ("basic", "mw", f"{target}_ci_lower", f"{target}_ci_upper"):
        df[col] = fs.loc[df.index, col]
    df = df.dropna()

    for label, sub in (("basic amine", df[df["basic"]]), ("no basic N", df[~df["basic"]])):
        if len(sub) < 30:
            continue
        resid = sub["pred"] - sub["y"]
        print(
            f"{iso if label.startswith('basic') else '':<9}{label:<14}{len(sub):>6}"
            f"{spearmanr(sub['y'], sub['pred']).statistic:>11.3f}"
            f"{pearsonr(sub['y'], sub['pred']).statistic:>9.3f}"
            f"{resid.abs().mean():>8.3f}{resid.mean():>+8.3f}{sub['mw'].mean():>7.0f}"
        )

print(f"\n{'isoform':<9}{'band':>10}{'n':>6}{'pred - ci_hi':>14}{'above':>8}{'below':>8}{'inside':>8}")
for iso in ISOFORMS:
    target = f"{iso.lower()}_pic50_direct_inhibition"
    df = oof_average(MEMBERS[iso], target)
    df["lo"] = fs.loc[df.index, f"{target}_ci_lower"]
    df["hi"] = fs.loc[df.index, f"{target}_ci_upper"]
    df = df.dropna()
    for label, sub in (("<4.0", df[df["y"] < 4.0]), (">=4.0", df[df["y"] >= 4.0])):
        if sub.empty:
            continue
        above = (sub["pred"] > sub["hi"]).mean()
        below = (sub["pred"] < sub["lo"]).mean()
        print(
            f"{iso if label == '<4.0' else '':<9}{label:>10}{len(sub):>6}"
            f"{np.mean(sub['pred'] - sub['hi']):>+14.2f}{above:>8.1%}{below:>8.1%}{1 - above - below:>8.1%}"
        )
