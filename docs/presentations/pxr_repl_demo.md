# Demo: REPL + Bosco on the PXR challenge

A recorded walkthrough. The spine is the handoff — **REPL to explore, Bosco to build, REPL to
inspect what it built** — repeated four times, on work that is real rather than staged.
Edited for pace; say so on screen.

**Why PXR.** One target (`pec50`), and a held-out set with real labels: OpenADMET revealed the
253-compound phase-1 set, so every score is honest and nothing waits on a leaderboard. The
contest is closed, so nothing here is a competitive claim.

**Everything is ad-hoc.** Bosco creates the artifacts in-session. No scripts, no
`pipelines.json`. That is the point being demonstrated: the artifacts are addressable the
moment they exist, by name, from the REPL — no export step, no copy-paste, no notebook.

**Naming.** Everything built on camera takes a `-demo` suffix. `openadmet_pxr_f1` and
`pxr-reg-chemprop-phase1` already exist, and the latter carries the `pxr_phase1_test` capture
the numbers below came from.

---

## 1. Public data to a local model — REPL → Bosco → REPL

**REPL.** Pull the data and look at it. No credentials needed for either of these:

```python
train = PublicData().get("comp_chem/openadmet/pxr/training/main")            # 4,139
test  = PublicData().get("comp_chem/openadmet/pxr/testing/phase1_unblinded") # 253
```

**Bosco.** Hand off the part that carries the traps: *"mark the split, compute 2D descriptors,
build a local FeatureSet and a local XGBoost model holding out the phase-1 rows."*

That one request covers `compute_descriptors`, picking a feature list that excludes
`molecule_name` / `smiles` / `pec50` / its error and interval columns / the `split` marker, and
wiring the holdout. Doing it by hand is where the mistakes live.

**REPL.** The local model is on disk and addressable by name. Inspect it, check the feature
list it chose, pull the out-of-fold predictions.

2D only — `compute_descriptors` is RDKit plus Mordred and is fast. The 3D blocks need conformer
generation and xTB, which is not a live-loop operation.

Workbench models are a 5-fold CV ensemble, so one call yields both the out-of-fold predictions
and the fold-averaged predictor.

## 2. Residual plots — out of fold

**REPL.** Interactive, you drive. Full cross-fold residuals set the visual baseline.

## 3. Residual plots — phase-1 held out

**REPL.** The same plot on the 253 revealed compounds. **This does not go the way the room
expects, and that is the point.** Measured on the existing chemprop phase-1 model:

| capture | n | MAE | R² | Pearson | Spearman | k = sd(pred)/sd(true) |
|---|---|---|---|---|---|---|
| `full_cross_fold` | 4,392 | 0.490 | **0.615** | 0.788 | 0.751 | 0.854 |
| `pxr_phase1_test` | 253 | **0.472** | 0.527 | 0.752 | **0.797** | **0.630** |

Held-out MAE and Spearman are *better* than cross-fold. R² is worse anyway. Let someone ask
why — the answer is that R² fell while the ordering improved, and the whole cause is the spread
ratio collapsing from 0.854 to 0.630. The model ranks the held-out set well and predicts into
two thirds of its real range. In the plot that is systematic slope, not scatter.

**Calibration, if you take it here.** Affine: shift and scale, no retraining, ranks untouched.
The optimum is `k = ρ`, not `k = 1` — a model correlating at ρ should be ρ times as wide as
reality, because shrinking toward the mean is the right response to uncertainty. Applied to the
phase-1 predictions, R² goes 0.527 → 0.565, landing on its ρ² ceiling, Spearman bit-identical.

Frame it as a deployment defect: a model predicting into two thirds of the range under-calls
the extremes, and MAE never shows you that. Murphy (1988) if anyone wants the reference. Never
mention leaderboards — this is measured against revealed labels, which is the whole reason it
is defensible.

## 4. Where the big residuals come from — REPL → Bosco → REPL

**REPL.** Point at the outliers. Novel chemistry, or activity cliffs?

**Bosco.** Ask it. The question already has columns attached — UQ v1 fits its error model on
exactly `[prediction, prediction_std, knn_distance, knn_target_std, local_pred_gap]`.
`knn_distance` is novel chemistry: nothing similar in training. `knn_target_std` and
`local_pred_gap` are the cliff signal: close neighbours that disagree, or a prediction drifted
from what its neighbourhood supports.

**REPL.** Look at the compounds it separated, and their neighbours.

This is the beat where Bosco is doing analysis rather than plumbing, which is a different and
better argument for it.

**Trap: do not plot confidence against error on `full_cross_fold`.** For a model built before
2026-09-01 the cross-fold `confidence` came from the error model scoring its own training rows,
so the correlation looks far better than it is. `pxr_phase1_test` is genuine held-out inference
and is safe.

## 5. The same thing as cloud artifacts — Bosco

**Bosco.** *"Now build this for real: a FeatureSet and a chemprop model, `-demo` suffix, same
253 rows held out, then deploy it."*

Ad-hoc, in session. The FeatureSet build is quicker than it looks — `InferenceCache` keys the
feature endpoint by SMILES, so the expensive 3D leg is already cached.

Chemprop training is the natural cut point. Kick it, cut, return to a finished endpoint.

Narrate over the cut: this is the artifact chain, not a notebook. DataSource → FeatureSet →
Model → Endpoint, each registered, each carrying its own metrics and captures.

## 6. The dashboard

Where Workbench earns its keep, and the reason the cloud half exists in this demo. The model
arrives as a tracked artifact with metrics, inference captures, UQ columns and plots attached —
shareable, and still there next week. Nothing was written to a script to get it there.

## 7. Chemprop vs local XGBoost — Bosco → REPL

**Bosco.** *"Score both on the phase-1 rows and compare."*

**REPL.** Read the table. Three things to say:

- Quote the held-out numbers, not cross-fold. The table in §3 is why.
- Quote Pearson or Spearman for a modelling difference. R² and MAE move with calibration, which
  is not a model property — a badly calibrated model with better ordering is the better model,
  and §3 is the proof.
- It is not a clean architecture comparison. Local XGBoost reads locally-computed 2D
  descriptors; cloud Chemprop reads SMILES and learns its own representation. Two variables,
  not one. Say so rather than declaring a winner.

---

## Notes for the take

- The local half needs no credentials at all. Worth saying out loud at §1.
- Nothing here needs ST-RAE. The credible intervals are in `PublicData` under
  `comp_chem/openadmet/pxr/testing/phase1_unblinded` if you want them.
- 253 rows is small. A Pearson difference under ~0.05 between two models on that set is not a
  result; say so rather than reading it as one.
- The PXR contest is closed. No live-standing claims.
