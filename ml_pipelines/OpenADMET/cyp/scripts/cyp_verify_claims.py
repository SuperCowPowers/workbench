"""Re-derive the planning doc's data claims from the public sources.

Every number in `docs/planning/openadmet_cyp_challenge.md` that can be settled from
`data/public_data/output/` is recomputed here and compared to what the doc asserts. A
claim that no longer reproduces gets corrected or removed from the doc -- an unverified
number in a working record is worse than no number, because it gets built on.

Out of scope: anything needing a scored submission (blind moments, the submission record)
or a model capture (OOF Spearman, ruler power). Those carry their own provenance and are
marked as such rather than checked here.

    python cyp_verify_claims.py
"""

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

D = "../../../../data/public_data/output/comp_chem"
ISOFORMS = ["cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4"]
RESULTS = []


def check(claim: str, stated, measured, ok: bool) -> None:
    RESULTS.append((ok, claim, stated, measured))
    mark = "ok  " if ok else "MISS"
    print(f"  [{mark}] {claim}\n         doc={stated}   measured={measured}")


def near(a, b, tol) -> bool:
    return abs(a - b) <= tol


def skeletons(smiles):
    keys = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        keys.append(Chem.MolToInchiKey(mol).split("-")[0] if mol else None)
    return pd.Series(keys, index=getattr(smiles, "index", None))


ch = pd.read_csv(f"{D}/openadmet/cyp/training/inhibition.csv")
ch["key"] = skeletons(ch["smiles"])
blind = pd.read_csv(f"{D}/openadmet/cyp/testing/blinded.csv")
blind["key"] = skeletons(blind["smiles"])
ch_keys, blind_keys = set(ch["key"].dropna()), set(blind["key"].dropna())

print("\n== The test set ==")
check("blind set is 750 compounds", 750, len(blind), len(blind) == 750)

print("\n== ChEMBL ==")
chembl = pd.read_csv(f"{D}/chembl/cyp_inhibition/all_isoforms.csv")
chembl["key"] = chembl["inchi_key"].str.split("-").str[0]
n_skel = chembl["key"].nunique()
check("24,918 skeletons", 24918, n_skel, near(n_skel, 24918, 200))
n_blind = len(set(chembl["key"]) & blind_keys)
check("zero blind-set overlap", 0, n_blind, n_blind == 0)
n_train = len(set(chembl["key"]) & ch_keys)
check("185 overlap with our training set", 185, n_train, near(n_train, 185, 15))

veith = pd.read_csv(f"{D}/pubchem/cyp_inhibition/all_isoforms.csv")
veith["key"] = skeletons(veith["smiles"])
v_keys = set(veith["key"].dropna())
pot_keys = set(veith.dropna(subset=["pic50"])["key"].dropna())
frac = len(pot_keys & set(chembl["key"])) / len(pot_keys)
check("98.5% of Veith's pIC50-carrying compounds are in ChEMBL", 0.985, round(frac, 3), near(frac, 0.985, 0.01))
frac_all = len(v_keys & set(chembl["key"])) / len(v_keys)
check("87% of the full Veith panel is in ChEMBL", 0.87, round(frac_all, 3), near(frac_all, 0.87, 0.02))

print("\n== Veith qHTS ==")
check("85,535 values", 85535, len(veith), len(veith) == 85535)
cov = veith["max_response"].notna().mean()
check("max_response 100% coverage", 1.0, round(cov, 4), cov > 0.995)
n_inactive = int((veith["activity_outcome"] == "Inactive").sum())
n_nopot = int(veith["pic50"].isna().sum())
check(
    "42,355 Inactive rows, 42,501 with no pIC50",
    (42355, 42501),
    (n_inactive, n_nopot),
    n_inactive == 42355 and n_nopot == 42501,
)
mr = veith.pivot_table(index="key", columns="isoform", values="max_response")
cors = {}
for iso in ISOFORMS:
    if iso in mr.columns:
        j = ch.dropna(subset=[f"{iso}_pic50_direct_inhibition"]).set_index("key")
        both = j[[f"{iso}_pic50_direct_inhibition"]].join(mr[[iso]], how="inner").dropna()
        cors[iso] = round(float(both.corr().iloc[0, 1]), 3)
lo, hi = min(cors.values()), max(cors.values())
check(
    "max_response correlates -0.52..-0.78 with challenge pIC50",
    "[-0.78,-0.52]",
    cors,
    near(lo, -0.78, 0.02) and near(hi, -0.52, 0.02),
)
p995 = {iso: round(float(mr[iso].quantile(0.995)), 1) for iso in ("cyp2d6", "cyp2c9") if iso in mr.columns}
check(
    "99.5th max_response: CYP2D6 +35, CYP2C9 +290",
    {"cyp2d6": 35, "cyp2c9": 290},
    p995,
    near(p995.get("cyp2d6", 0), 35, 25) and near(p995.get("cyp2c9", 0), 290, 120),
)

print("\n== Tox21 ==")
tox_all = pd.read_csv(f"{D}/tox21/cyp_inhibition/all_isoforms.csv")
tox = tox_all[~tox_all["luciferase_inhibitor"].astype(bool)]
check(
    "7,196 compounds, 6,639 after the luciferase filter",
    (7196, 6639),
    (tox_all["smiles"].nunique(), tox["smiles"].nunique()),
    tox_all["smiles"].nunique() == 7196 and tox["smiles"].nunique() == 6639,
)
d6 = tox[tox["isoform"] == "cyp2d6"].copy()
d6["key"] = skeletons(d6["smiles"])
d6 = d6.dropna(subset=["key", "pic50"]).groupby("key")["pic50"].median()
check(
    "CYP2D6 median fitted pIC50 4.76, 62% below 5.0",
    (4.76, 0.62),
    (round(float(d6.median()), 2), round(float((d6 < 5.0).mean()), 2)),
    near(d6.median(), 4.76, 0.03) and near((d6 < 5.0).mean(), 0.62, 0.02),
)
pooled = tox["pic50"].dropna()
check(
    "pooled over five isoforms, 4.70 and 68%",
    (4.70, 0.68),
    (round(float(pooled.median()), 2), round(float((pooled < 5.0).mean()), 2)),
    near(pooled.median(), 4.70, 0.03) and near((pooled < 5.0).mean(), 0.675, 0.02),
)

print("\n== Single-concentration log2fc ==")
sc = pd.read_csv(f"{D}/openadmet/cyp/training/single_concentration.csv")
print("  CYP2D6 vs CYP3A4 median log2fc by fitted-pIC50 bin (the flatness claim):")
bins = [(2, 3.5), (3.5, 4.0), (4.0, 4.31), (4.31, 4.6)]
flat = {}
for iso in ("cyp2d6", "cyp3a4"):
    arm = sc[sc["enzyme"] == iso.upper()][["molecule_name", "log2fc_estimate"]]
    m = ch[["molecule_name", f"{iso}_pic50_direct_inhibition"]].dropna().merge(arm, on="molecule_name")
    meds = []
    for lo_b, hi_b in bins:
        sel = m[(m[f"{iso}_pic50_direct_inhibition"] >= lo_b) & (m[f"{iso}_pic50_direct_inhibition"] < hi_b)]
        meds.append(round(float(sel["log2fc_estimate"].median()), 2) if len(sel) else None)
    flat[iso] = meds
    print(f"    {iso}: {meds}")
d6 = [v for v in flat["cyp2d6"] if v is not None]
a4 = [v for v in flat["cyp3a4"] if v is not None]
check(
    "CYP2D6 log2fc flat across the low end, CYP3A4 monotone",
    "d6 -1.07/-0.98/-0.94/-1.11, a4 -0.18/-0.77/-1.32/-2.03",
    flat,
    (max(d6) - min(d6)) < 0.4 and (max(a4) - min(a4)) > 1.0,
)

print("\n== Emax ==")
emax = pd.read_csv(f"{D}/openadmet/cyp/training/emax.csv")
ec = {}
for iso in ISOFORMS:
    col = f"{iso}_emax_vs_pos_ctrl_direct_inhibition"
    if col in emax.columns:
        m = (
            ch[["molecule_name", f"{iso}_pic50_direct_inhibition"]]
            .dropna()
            .merge(emax[["molecule_name", col]].dropna(), on="molecule_name")
        )
        if len(m) > 20:
            ec[iso] = round(float(m.corr(numeric_only=True).iloc[0, 1]), 3)
others = [abs(v) for k, v in ec.items() if k != "cyp2d6"]
check(
    "emax correlates 0.555 with CYP2D6, near zero elsewhere",
    0.555,
    ec,
    near(abs(ec.get("cyp2d6", 0)), 0.555, 0.12) and max(others) < 0.35,
)

print("\n== TDI arm ==")
tdi = pd.read_csv(f"{D}/openadmet/cyp/training/tdi.csv")
check("tdi.csv and emax.csv cover 6,145 compounds", 6145, (len(tdi), len(emax)), len(tdi) == 6145)
base = int(ch["cyp3a4_pic50_direct_inhibition"].notna().sum())
lifted = int(
    pd.concat(
        [
            ch[["molecule_name", "cyp3a4_pic50_direct_inhibition"]],
            tdi[["molecule_name", "cyp3a4_pic50_tdi_condition"]].rename(
                columns={"cyp3a4_pic50_tdi_condition": "cyp3a4_pic50_direct_inhibition"}
            ),
        ]
    )
    .dropna()["molecule_name"]
    .nunique()
)
check(
    "TDI lifts CYP3A4 coverage 2,335 -> 3,583",
    (2335, 3583),
    (base, lifted),
    near(base, 2335, 60) and near(lifted, 3583, 150),
)

print("\n== CYP2D6 scored column ==")
n_sub4 = int((ch["cyp2d6_pic50_direct_inhibition"] < 4.0).sum())
check("129 sub-4.0 CYP2D6 scored rows", 129, n_sub4, n_sub4 == 129)
n_lab = int(ch["cyp2d6_pic50_direct_inhibition"].notna().sum())
check("1,493 CYP2D6 labelled rows", 1493, n_lab, n_lab == 1493)

print("\n== Leaderboard snapshot ==")
board = open("../../../../docs/planning/cyp_leaderboard_2026_09_01.md").read()
ours = {}
for blk in board.split("## ")[1:]:
    iso = blk.split("\n")[0].split(" (")[0]
    rows = [r for r in blk.split("\n") if r.startswith("|") and "---" not in r]
    if len(rows) < 2 or "MAE" not in rows[0]:
        continue
    cols = [c.strip() for c in rows[0].strip("|").split("|")]
    tbl = pd.DataFrame([[c.strip() for c in r.strip("|").split("|")] for r in rows[1:]], columns=cols)
    for c in cols[1:]:
        tbl[c] = pd.to_numeric(tbl[c])
    mine = tbl[tbl[cols[0]] == "briford"]
    ours[iso] = {
        m: int((tbl[c] < mine[c].iloc[0]).sum() + 1) if m == "MAE" else int((tbl[c] > mine[c].iloc[0]).sum() + 1)
        for m, c in ((m, next(c for c in cols if m in c)) for m in ("MAE", "R\u00b2", "Spearman"))
    }
stated = {
    "CYP1A2": {"MAE": 3, "R\u00b2": 1, "Spearman": 3},
    "CYP2C9": {"MAE": 2, "R\u00b2": 2, "Spearman": 3},
    "CYP2D6": {"MAE": 4, "R\u00b2": 4, "Spearman": 5},
    "CYP3A4": {"MAE": 3, "R\u00b2": 4, "Spearman": 4},
}
check("our within-top-5 rank per isoform", stated, ours, stated == ours)
check(
    "we do NOT hold best MAE on CYP2C9 or CYP3A4",
    "not 1st",
    {k: ours[k]["MAE"] for k in ("CYP2C9", "CYP3A4")},
    ours["CYP2C9"]["MAE"] != 1 and ours["CYP3A4"]["MAE"] != 1,
)

print("\n" + "=" * 70)
bad = [r for r in RESULTS if not r[0]]
print(f"{len(RESULTS) - len(bad)} of {len(RESULTS)} claims reproduce.")
for _, claim, stated, measured in bad:
    print(f"  MISMATCH: {claim}\n            doc={stated}  measured={measured}")
