"""Build synthetic LogP/LogD datasets for chemprop multi-task implementation validation.

Real-data experiments showed no MT lift, plausibly because our LogP/LogD overlap
has Pearson 0.967 (auxiliary nearly redundant with primary). To isolate whether
the chemprop multi-task wiring is correct independent of data quality, we generate
multiple synthetic LogP recipes — fully fake, no chemistry-correctness constraint —
on mid-tier compounds (chemistry distinct from the LogD set).

Four recipes, increasing in "guaranteed informativeness":

  Recipe A — pure Crippen.MolLogP. Atom-additive lipophilicity. Pearson with
             real LogD on this dataset ~0.39.
  Recipe B — Crippen + aromatic-fraction + rotatable-bond terms. Forces the GNN
             to learn additional structural features. Pearson ~0.42.
  Recipe C — Random Forest predictions of LogD trained on the full LogD corpus.
             Synthetic LogP is "predicted LogD on extended chemistry" — a
             deterministic function of structure that captures LogD's
             descriptor-level structure-property relationships.
  Recipe D — Real LogD on extended chemistry, stuffed into the logp column.
             The auxiliary is literally LogD on 3,199 LogD-corpus compounds the
             primary head never sees. MT under this recipe has access to ~4x more
             LogD-relevant supervision than the 1k-compound ST baseline. Cross-
             the-board lift is essentially guaranteed if MT works at all; if it
             doesn't beat ST here, the wiring is broken.

Inputs (must already exist):
    output/experiments/logp_logd_overlap_03_07.csv   (band-strict mid-tier)
    output/logd/logd_all.csv                         (full LogD corpus)

Outputs:
    output/synthetic/multi_task/log_d.csv          (1,000 real LogD rows)
    output/synthetic/multi_task/log_p.csv          (Recipe A)
    output/synthetic/multi_task/log_p_blended.csv  (Recipe B)
    output/synthetic/multi_task/log_p_strong.csv   (Recipe C)
    output/synthetic/multi_task/log_p_real.csv     (Recipe D, real LogD on extended)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output"
MID_TIER_PATH = OUTPUT_DIR / "experiments" / "logp_logd_overlap_03_07.csv"
LOGD_CORPUS_PATH = OUTPUT_DIR / "logd" / "logd_all.csv"
SYNTH_DIR = OUTPUT_DIR / "synthetic" / "multi_task"

DEFAULT_N = 1000

# Descriptors used by the Recipe C teacher model. Kept simple/standard so a GNN
# given enough capacity can learn to reproduce each one from raw graphs.
DESCRIPTOR_FUNCS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.NumAromaticRings,
    Descriptors.NumAliphaticRings,
    Descriptors.NumHeteroatoms,
    Descriptors.TPSA,
    Descriptors.RingCount,
    Descriptors.FractionCSP3,
    Descriptors.HeavyAtomCount,
]


def _descriptors(smiles: str) -> list[float] | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else [f(mol) for f in DESCRIPTOR_FUNCS]


def fit_logd_teacher(logd_corpus_path: Path):
    """Recipe C teacher: Random Forest fit on RDKit descriptors -> real LogD.

    Returns a callable: smiles -> predicted_logd. By construction, predictions are
    a deterministic function of structure that captures LogD's descriptor-level
    structure-property relationships from the full LogD corpus.
    """
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv(logd_corpus_path)
    rows = [(s, l) for s, l in zip(df["smiles"], df["logd"]) if _descriptors(s) is not None]
    X = np.array([_descriptors(s) for s, _ in rows])
    y = np.array([l for _, l in rows])
    log.info(f"Fitting Recipe C teacher (RandomForest) on {len(X):,} LogD compounds...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X, y)

    def predict(smiles: str) -> float | None:
        d = _descriptors(smiles)
        return None if d is None else float(model.predict([d])[0])

    return predict


def crippen_logp(smiles: str) -> float | None:
    """Recipe A: pure Crippen.MolLogP — atom-additive lipophilicity."""
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Crippen.MolLogP(mol)


def crippen_blended_logp(smiles: str) -> float | None:
    """Recipe B: Crippen + aromatic-fraction + rotatable-bond term.

    Forces the GNN to learn additional structural features beyond atom-additive
    lipophilicity: aromatic atom counting and rotatable bond counting. Coefficients
    chosen to give a slightly higher Pearson with experimental LogD than pure
    Crippen on this dataset (~0.42 vs ~0.39).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    crippen = Crippen.MolLogP(mol)
    aro_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    heavy = mol.GetNumHeavyAtoms()
    aro_frac = aro_atoms / max(heavy, 1)
    nrot = Descriptors.NumRotatableBonds(mol)
    return crippen + 2.0 * aro_frac - 0.05 * nrot


def main(n_per_set: int, seed: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not MID_TIER_PATH.exists():
        log.error(f"Source file missing: {MID_TIER_PATH}\n" f"Run build_logp_logd_overlap.py first to produce it.")
        sys.exit(1)
    if not LOGD_CORPUS_PATH.exists():
        log.error(f"Source file missing: {LOGD_CORPUS_PATH}")
        sys.exit(1)

    mid = pd.read_csv(MID_TIER_PATH)
    log.info(f"Loaded mid-tier source: {len(mid):,} rows from {MID_TIER_PATH.name}")

    # Pools: LogD-only and LogP-only rows from the band-strict mid tier.
    logd_pool = mid[mid["logd"].notna() & mid["logp"].isna()][["smiles", "logd"]].reset_index(drop=True)
    logp_pool = mid[mid["logp"].notna() & mid["logd"].isna()][["smiles"]].reset_index(drop=True)
    log.info(f"  LogD-only pool: {len(logd_pool):,}")
    log.info(f"  LogP-only pool: {len(logp_pool):,}  (Tanimoto 0.3-0.7 to LogD)")

    if len(logd_pool) < n_per_set or len(logp_pool) < n_per_set:
        log.error(f"Pool too small to sample {n_per_set} per set")
        sys.exit(1)

    # ---- LogD set: real labels, random sample ----
    logd_set = logd_pool.sample(n=n_per_set, random_state=seed).reset_index(drop=True)
    logd_set.insert(0, "id", [f"logd_{i}" for i in range(len(logd_set))])
    log.info(
        f"LogD set: {len(logd_set):,} rows  "
        f"(real logd range [{logd_set['logd'].min():.2f}, {logd_set['logd'].max():.2f}])"
    )

    # ---- Recipe C: fit teacher model on full LogD corpus ----
    teacher = fit_logd_teacher(LOGD_CORPUS_PATH)

    # ---- LogP sets A, B, C: mid-tier chemistry, synthetic labels ----
    logp_smiles = logp_pool.sample(n=n_per_set, random_state=seed + 1).reset_index(drop=True)

    logp_set_a = logp_smiles.copy()
    logp_set_a["logp"] = logp_set_a["smiles"].apply(crippen_logp)

    logp_set_b = logp_smiles.copy()
    logp_set_b["logp"] = logp_set_b["smiles"].apply(crippen_blended_logp)

    logp_set_c = logp_smiles.copy()
    logp_set_c["logp"] = logp_set_c["smiles"].apply(teacher)

    # ---- Recipe D: extended LogD compounds with real LogD as the "synthetic" logp ----
    # Pull every LogD compound NOT in the primary set; their real LogD values become
    # the synthetic logp. The auxiliary head is supervised by real LogD on chemistry
    # the primary head never sees -> ~4x more LogD-relevant supervision than ST.
    primary_smiles = set(logd_set["smiles"])
    extended = logd_pool[~logd_pool["smiles"].isin(primary_smiles)].reset_index(drop=True)
    logp_set_d = extended.rename(columns={"logd": "logp"})[["smiles", "logp"]]

    for name, df in [
        ("Recipe A (Crippen)", logp_set_a),
        ("Recipe B (blended)", logp_set_b),
        ("Recipe C (RF teacher)", logp_set_c),
        ("Recipe D (real LogD on extended)", logp_set_d),
    ]:
        n_invalid = df["logp"].isna().sum()
        if n_invalid > 0:
            log.warning(f"{name}: dropping {n_invalid} rows where computation failed")
            df.dropna(subset=["logp"], inplace=True)
            df.reset_index(drop=True, inplace=True)

    for df in (logp_set_a, logp_set_b, logp_set_c, logp_set_d):
        df.insert(0, "id", [f"logp_{i}" for i in range(len(df))])

    log.info(
        f"LogP set A (pure Crippen):                {len(logp_set_a):,} rows  "
        f"range [{logp_set_a['logp'].min():.2f}, {logp_set_a['logp'].max():.2f}]"
    )
    log.info(
        f"LogP set B (blended):                     {len(logp_set_b):,} rows  "
        f"range [{logp_set_b['logp'].min():.2f}, {logp_set_b['logp'].max():.2f}]"
    )
    log.info(
        f"LogP set C (RF teacher):                  {len(logp_set_c):,} rows  "
        f"range [{logp_set_c['logp'].min():.2f}, {logp_set_c['logp'].max():.2f}]"
    )
    log.info(
        f"LogP set D (real LogD on extended):       {len(logp_set_d):,} rows  "
        f"range [{logp_set_d['logp'].min():.2f}, {logp_set_d['logp'].max():.2f}]"
    )

    # ---- Sanity check: each recipe's Pearson with real LogD on the LogD set ----
    valid_logd = logd_set["logd"].notna().values
    real_logd = logd_set.loc[valid_logd, "logd"].astype(float).values

    cr_a = logd_set["smiles"].apply(crippen_logp).astype(float).values[valid_logd]
    cr_b = logd_set["smiles"].apply(crippen_blended_logp).astype(float).values[valid_logd]
    cr_c = logd_set["smiles"].apply(teacher).astype(float).values[valid_logd]
    r_a = np.corrcoef(cr_a, real_logd)[0, 1]
    r_b = np.corrcoef(cr_b, real_logd)[0, 1]
    r_c = np.corrcoef(cr_c, real_logd)[0, 1]

    log.info("")
    log.info("Sanity check: synthetic-aux Pearson with real LogD on the LogD set")
    log.info(f"  Recipe A (pure Crippen):       r = {r_a:.3f}")
    log.info(f"  Recipe B (Crippen + aro/rot):  r = {r_b:.3f}")
    log.info(f"  Recipe C (RF teacher):         r = {r_c:.3f}  (training fit; teacher saw these)")
    log.info("  Recipe D (real LogD on ext.):  r = 1.000  (by construction; aux IS real LogD)")
    log.info("  Reference: RTlogD paper        r = 0.628  -> 3% RMSE MT lift")
    log.info("")
    log.info("Recipe D is the cross-the-board guaranteed-lift recipe: the auxiliary head")
    log.info("is supervised by real LogD on 3,199 LogD-corpus compounds the primary never")
    log.info("sees. MT-with-D has access to ~4x more LogD-relevant supervision than ST.")
    log.info("If MT-D doesn't beat ST on every metric, the chemprop multi-task wiring")
    log.info("is broken.")

    # ---- Write outputs ----
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    log_d_out = SYNTH_DIR / "log_d.csv"
    log_p_out = SYNTH_DIR / "log_p.csv"
    log_p_blended_out = SYNTH_DIR / "log_p_blended.csv"
    log_p_strong_out = SYNTH_DIR / "log_p_strong.csv"
    log_p_real_out = SYNTH_DIR / "log_p_real.csv"
    logd_set[["id", "smiles", "logd"]].to_csv(log_d_out, index=False)
    logp_set_a[["id", "smiles", "logp"]].to_csv(log_p_out, index=False)
    logp_set_b[["id", "smiles", "logp"]].to_csv(log_p_blended_out, index=False)
    logp_set_c[["id", "smiles", "logp"]].to_csv(log_p_strong_out, index=False)
    logp_set_d[["id", "smiles", "logp"]].to_csv(log_p_real_out, index=False)
    log.info("")
    log.info(f"Saved -> {log_d_out}")
    log.info(f"Saved -> {log_p_out}")
    log.info(f"Saved -> {log_p_blended_out}")
    log.info(f"Saved -> {log_p_strong_out}")
    log.info(f"Saved -> {log_p_real_out}")
    log.info("")
    log.info("Next steps:")
    log.info("  1. Upload to public data:")
    log.info("       comp_chem/synthetic/multi_task/log_d")
    log.info("       comp_chem/synthetic/multi_task/log_p")
    log.info("       comp_chem/synthetic/multi_task/log_p_blended")
    log.info("       comp_chem/synthetic/multi_task/log_p_strong")
    log.info("       comp_chem/synthetic/multi_task/log_p_real")
    log.info("  2. Run examples/models/chemprop_logp_logd_synthetic.py to train and compare.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-n",
        "--n-per-set",
        type=int,
        default=DEFAULT_N,
        help=f"Rows per dataset (default: {DEFAULT_N})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(args.n_per_set, args.seed)
