"""Could a public source be pooled into a scored column, rather than kept as its own head?

An auxiliary head keeps its own scale, which is what makes it safe across assays and also
what stops low-range information reaching the scored output. Pooling puts public values in
the scored column instead, and that needs a calibrated map from the public assay's scale to
the challenge's. This scores whether such a map is measurable, before anyone fits one.

Four things decide it, and the third is the one that has already cost us:

  anchors     compounds both assays measured. Without them there is nothing to fit.
  agreement   cross-assay Spearman on the anchors. Below ~0.6 no shift will rescue it.
  reach       what share of the source's values fall inside the range the anchors span.
              A map fitted on potent compounds and applied to weak ones is extrapolation,
              which is how the Tox21 attempt failed.
  residual    scatter left after the shift, against the challenge's own label noise
              (median std 0.07). Pooling at a residual far above that adds rows whose
              labels are noisier than the ones already there.

`E[challenge | public]` is the estimator, not the structural fit: the correction is applied
given a public reading, so the regression runs that way round.

    python cyp_pooling_precheck.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

D = "../../../../data/public_data/output/comp_chem"
ISOFORMS = ["cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4"]
LABEL_NOISE = 0.07  # median challenge pIC50 std
MIN_AGREEMENT = 0.6


def skeletons(smiles: pd.Series) -> pd.Series:
    keys = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        keys.append(Chem.MolToInchiKey(mol).split("-")[0] if mol else None)
    return pd.Series(keys, index=smiles.index)


def public_frames() -> dict:
    """Per-source, per-isoform pIC50 keyed on skeleton."""
    out = {}
    chembl = pd.read_csv(f"{D}/chembl/cyp_inhibition/all_isoforms.csv")
    chembl["key"] = chembl["inchi_key"].str.split("-").str[0]
    out["chembl"] = {i: chembl[["key", f"{i}_pic50"]].rename(columns={f"{i}_pic50": "public"}) for i in ISOFORMS}

    for name, path, lucif in (
        ("veith", f"{D}/pubchem/cyp_inhibition/all_isoforms.csv", False),
        ("tox21", f"{D}/tox21/cyp_inhibition/all_isoforms.csv", True),
    ):
        df = pd.read_csv(path)
        if lucif:
            df = df[~df["luciferase_inhibitor"].astype(bool)]
        df["key"] = skeletons(df["smiles"])
        out[name] = {i: df[df["isoform"] == i][["key", "pic50"]].rename(columns={"pic50": "public"}) for i in ISOFORMS}
    return out


def main() -> None:
    ch = pd.read_csv(f"{D}/openadmet/cyp/training/inhibition.csv")
    ch["key"] = skeletons(ch["smiles"])
    ch = ch.dropna(subset=["key"]).drop_duplicates("key")

    print(
        f"{'source':7s} {'isoform':8s} {'anchors':>8s} {'<5.0':>5s} {'rho':>6s} {'offset':>7s} "
        f"{'resid':>6s} {'reach':>6s}  verdict"
    )
    for source, frames in public_frames().items():
        for iso in ISOFORMS:
            pub = frames[iso].dropna(subset=["key", "public"]).groupby("key", as_index=False)["public"].median()
            col = f"{iso}_pic50_direct_inhibition"
            anchors = ch[["key", col]].dropna().merge(pub, on="key")
            n = len(anchors)
            if n < 10:
                print(
                    f"{source:7s} {iso:8s} {n:8,} {'--':>5s} {'--':>6s} {'--':>7s} {'--':>6s} {'--':>6s}  "
                    f"too few anchors"
                )
                continue
            rho = float(anchors[[col, "public"]].corr(method="spearman").iloc[0, 1])
            slope, icept = np.polyfit(anchors["public"], anchors[col], 1)
            resid = float((anchors[col] - (slope * anchors["public"] + icept)).std(ddof=1))
            offset = float((anchors[col] - anchors["public"]).mean())
            low = int((anchors["public"] < 5.0).sum())
            lo, hi = anchors["public"].min(), anchors["public"].max()
            reach = float(((pub["public"] >= lo) & (pub["public"] <= hi)).mean())

            if rho < MIN_AGREEMENT:
                verdict = f"no — agreement {rho:.2f} < {MIN_AGREEMENT}"
            elif low < 30:
                verdict = f"no — only {low} anchors below 5.0"
            elif resid > 5 * LABEL_NOISE:
                verdict = f"weak — residual {resid:.2f} vs {LABEL_NOISE} label noise"
            else:
                verdict = "candidate"
            print(
                f"{source:7s} {iso:8s} {n:8,} {low:5d} {rho:6.2f} {offset:+7.2f} {resid:6.2f} "
                f"{reach:5.0%}  {verdict}"
            )


if __name__ == "__main__":
    main()
