"""PXR phase-1 ablation: why the deployed TabICL + Monroe + readouts model trails the spike.

The spike (pxr_monroe_spike_phase1.py) scored RAE 0.542; the deployed model
(pxr-reg-tabicl-monroe-readout-phase1) scored 0.559. They differ in how many Monroe
columns TabICL gets (spike: top 360 by variance; deployed: all 720), seeds (spike: 3
averaged; deployed: 1), and the embedding itself (deployed: standardized SMILES and a
seeded conformer, read from openadmet_pxr_readout_monroe).

Every arm here uses the deployed embedding plus the two readouts, fit on the `train`
rows and scored on the 253 `phase1_test` rows:

  top100, top360, all720   the highest-variance Monroe columns (variance on train rows)
  pca100                   StandardScaler + PCA on the Monroe columns only (fit on train
                           rows); the readouts stay raw

Each arm is fit with SEEDS and scored two ways: the seed-averaged prediction, and the
mean single-seed RAE (the deployed model is a single seed). If top360 averaged lands near
0.542, the deployed embedding is cleared and the gap is columns and/or seeds.

Results (phase1_test, 3 seeds, paired RAE delta with 95% CI):

    arm               RAE    MAE    single seed     vs chemprop               vs chemprop_readout
    chemprop          0.591  0.472
    chemprop_readout  0.569  0.454
    deployed all720   0.559  0.446  (seed 42)
    top100            0.537  0.429  0.538 ± 0.005   -0.053 [-0.088, -0.017]   -0.032 [-0.062, -0.002]
    pca100            0.554  0.442  0.554 ± 0.005   -0.037 [-0.072, -0.004]   -0.015 [-0.052, +0.021]
    top360            0.540  0.431  0.541 ± 0.001   -0.050 [-0.084, -0.018]   -0.029 [-0.056, -0.003]
    all720            0.550  0.439  0.551 ± 0.001   -0.041 [-0.075, -0.007]   -0.019 [-0.046, +0.007]

top360 reproduces the spike (0.540 vs 0.542), so the deployed embedding is not the gap.
Seed averaging is worth ~0.001. Column count is the gap: all720 trails top360 by 0.010.
The deployed model (0.559) also trails this all720 fit (0.551) by 0.008, unexplained.

Arms run smallest first and print as they finish (all720 is the most memory on CPU).
Runs locally (not via ml_pipeline_launcher):  python pxr_monroe_ablation_phase1.py
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from workbench.api import FeatureSet, Model
from workbench.endpoints.inference import decompress_features
from workbench.endpoints.tabicl_utils import predict_with_std
from workbench.training.tabicl_core import fit_tabicl
from workbench.utils.metrics_utils import bootstrap_compare

fs_name = "openadmet_pxr_readout_monroe"
id_col, target = "molecule_name", "pec50"
READOUTS = ["lfc_8um_readout", "lfc_33um_readout"]
BASELINES = {"chemprop": "pxr-reg-chemprop-phase1", "chemprop_readout": "pxr-reg-chemprop-readout-phase1"}
HYPERPARAMETERS = {"n_estimators": 8, "batch_size": 1, "pca_components": None}
SEEDS = [0, 1, 2]


def rae(frame: pd.DataFrame) -> float:
    """Relative absolute error against the mean predictor (the challenge's headline)."""
    y = frame[target]
    return float((frame["prediction"] - y).abs().sum() / (y - y.mean()).abs().sum())


def mae(frame: pd.DataFrame) -> float:
    return float((frame["prediction"] - frame[target]).abs().mean())


# Data: the deployed embedding, expanded to 720 float columns
df = FeatureSet(fs_name).pull_dataframe()[[id_col, "monroe", target, "split"] + READOUTS]
df, monroe_cols = decompress_features(df, ["monroe"], ["monroe"])
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "phase1_test"].reset_index(drop=True)
print(f"{len(train)} train / {len(test)} phase1_test rows, {len(monroe_cols)} Monroe columns")


def frame(pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({target: test[target].to_numpy(), "prediction": pred}, index=test[id_col])


def captured(model_name: str) -> pd.DataFrame:
    """A deployed model's held-out `pxr_phase1_test` predictions, aligned to `test`."""
    held = Model(model_name).get_inference_predictions("pxr_phase1_test").set_index(id_col)["prediction"]
    pred = test[id_col].map(held).to_numpy()
    assert not np.isnan(pred).any(), f"{model_name} capture is missing phase1_test rows"
    return frame(pred)


def top_variance(k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The k highest-variance Monroe columns (variance on train rows) plus the readouts."""
    keep = list(train[monroe_cols].var().nlargest(k).index)
    return train[keep + READOUTS], test[keep + READOUTS]


def pca(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """n principal components of the Monroe columns only (fit on train rows) plus the raw readouts."""
    reducer = make_pipeline(StandardScaler(), PCA(n_components=n, random_state=0)).fit(train[monroe_cols])
    names = [f"pc_{i}" for i in range(n)]

    def project(rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(reducer.transform(rows[monroe_cols]), columns=names).assign(**rows[READOUTS])

    return project(train), project(test)


ARMS = {
    "top100": lambda: top_variance(100),
    "pca100": lambda: pca(100),
    "top360": lambda: top_variance(360),
    "all720": lambda: top_variance(720),
}

baselines = {name: captured(model_name) for name, model_name in BASELINES.items()}
deployed = captured("pxr-reg-tabicl-monroe-readout-phase1")
rows = [{"arm": name, "rae": rae(f), "mae": mae(f), "seeds": "deployed"} for name, f in baselines.items()]
rows.append({"arm": "deployed all720", "rae": rae(deployed), "mae": mae(deployed), "seeds": "1 (seed 42)"})

for arm, build in ARMS.items():
    X_train, X_test = build()
    per_seed = []
    for seed in SEEDS:
        model, _ = fit_tabicl({**HYPERPARAMETERS, "seed": seed}, X_train, train[target], cache=False)
        per_seed.append(predict_with_std(model, X_test)[0])
    averaged = frame(np.mean(per_seed, axis=0))
    single = [rae(frame(p)) for p in per_seed]
    row = {
        "arm": arm,
        "rae": rae(averaged),
        "mae": mae(averaged),
        "seeds": f"{len(SEEDS)} averaged",
        "single-seed rae": f"{np.mean(single):.3f} ± {np.std(single):.3f}",
    }
    for base, base_frame in baselines.items():
        cmp = bootstrap_compare(averaged, base_frame, rae)
        row[f"vs {base}"] = f"{cmp['delta']:+.3f} [{cmp['ci_lower']:+.3f}, {cmp['ci_upper']:+.3f}]"
    rows.append(row)
    summary = f"RAE {row['rae']:.3f} averaged, {row['single-seed rae']} single seed"
    print(f"{arm} ({X_train.shape[1]} features): {summary}", flush=True)

print(f"\nphase1_test, {len(test)} compounds (RAE: lower is better; deltas are paired RAE, 95% bootstrap)")
print(pd.DataFrame(rows).fillna("").to_string(index=False, float_format="%.3f"))
