# First-Class Hyperparameter Optimization (HPO) Support

## Principle (why this exists)

HPO is **ephemeral search**, not model publishing. `to_model()`/`to_endpoint()`
*publish* shared artifacts — discoverable by coworkers, in the model registry,
eligible for promotion. **HPO trials must never become Workbench
models/endpoints.** The correct shape: *search produces a config → Workbench
publishes only the one winner.* The search itself creates zero Workbench
artifacts.

## Converged design

HPO runs as **two phases inside a single training job**:

1. **Phase 1 — search.** Ray Tune / Optuna run many chemprop candidates
   *in-process*. Trials never call `to_model()`, never register. Objective is a
   held-out/OOD metric (below).
2. **Phase 2 — publish.** The winning config is trained as the final model and
   published through the normal `to_model()` path. Only the winner becomes an
   artifact.

This means **no new client API primitive.** HPO is just `to_model()` with an
`hpo` hyperparameter block:

```python
fs.to_model(
    model_framework=ModelFramework.CHEMPROP,
    target_column="pec50",
    feature_list=["smiles"],
    hyperparameters={
        "uq_version": "v1",
        "hpo": {
            "backend": "auto",        # optuna (local) | ray (offload) | auto
            "n_trials": 40,
            "max_parallel": 4,        # concurrent trials (Ray, offload)
            "metric": "holdout_mae",
            "search_space": { ... },  # see below
        },
    },
    validation_ids=holdout_ids,   # kept, marked `validation`, never trained
)
```

`pipelines.json` structure is **unchanged** (`fs → model:`). The non-negotiable
holds *by construction* — the trials are ephemeral and in-process; only phase 2
publishes.

### `hpo` block schema (draft)

| key | default | meaning |
|---|---|---|
| `backend` | `auto` | `optuna` serial (local), `ray` parallel + ASHA (offload), `auto` picks by env |
| `n_trials` | `40` | search budget |
| `max_parallel` | node GPUs | concurrent trials (Ray only) |
| `metric` | `holdout_mae` | objective; minimized |
| `search_space` | built-in | knob → range/choices (below) |

`uq_version` stays fixed at `v1` per repo convention.

### Objective + split

Default objective is **held-out / OOD MAE**. It rides on the landed
validation-set support: designate held-out rows with `validation_ids` (they're
kept in the training view, marked with a boolean `validation` column, and never
trained). Inside training, `workbench.training.validation.split_validation_set(df)`
routes them out of the train/CV path and scores them as an honest held-out set —
the chemprop template **already prints `holdout_mae`** on exactly these rows. So
each trial's objective is a metric the training path already emits; the HPO
harness reuses the same split + scoring rather than re-deriving a holdout. Because
we write our own objective function (chemprop's hpopt is a Python pattern, not a
fixed-`val_loss` CLI), this drops in directly.

### Search space (chemprop defaults)

`depth`, `message_hidden_dim`, `ffn_hidden_dim` (incl. **tapered lists** — the
reason we own the loop rather than use SageMaker AMT's scalar-only space),
`ffn_num_layers`, `dropout`, and the LR schedule (`warmup_epochs`, `init_lr`,
`max_lr`, `final_lr`), `max_epochs`. Ranges overridable per call.

## Compute vehicle

**One SageMaker training job**, reusing the existing `pytorch_chem` training
image (already pins chemprop 2.2.4 + CUDA). Rationale:

- A raw training job is **not** a Workbench model — `FeaturesToModel` launches
  the job *then* calls `create_and_register_model()`; HPO does the launch only.
  Registry-free by construction.
- chemprop stays pinned in **one** image (training-inference parity), and the
  winning config transfers faithfully because search ran in the identical env.
- **Parallelism = Ray Tune** across the node's GPUs, with **ASHA early-stopping**
  (kill weak trials early). Bounded by the instance's GPUs today; Ray Tune scales
  to a multi-node cluster later without changing the objective.

Rejected: SageMaker Automatic Model Tuning (unbounded horizontal fan-out, but
scalar-only search space + emitted-metric objective — loses tapered-FFN search
and full chemprop-ecosystem embrace). Also rejected (per task): a local Optuna
loop that calls `to_model()` per trial.

### Local vs offload — same script

`backend: auto` → **Optuna serial** on a laptop `--local` run, **Ray Tune
parallel** on the AWS offload. Same objective function; the runtime picks the
backend. Maps exactly onto the existing ml_pipeline "one script, runs local or
Batch" model.

### Instance sizing

`FeaturesToModel` bumps the chemprop training instance to **multi-GPU** (+ longer
max-runtime) *when `hpo` is set*, so Ray Tune has real parallel trials. This is a
training-instance selection keyed on `hpo` — **not** a new Batch tier. (The
CPU/Fargate Batch tiers only host the lightweight orchestration; the GPU work is
the training job.)

### Artifacts written

Alongside the published model, persist `best_config.json` and the trials
dataframe to S3 — an audit trail and a **deterministic re-publish escape hatch**
(rerun `to_model()` with the frozen config and no `hpo` block).

## Module + dependency architecture

The generated model script runs in **two** containers — training *and* the
(leaner) inference endpoint. `workbench.endpoints` is a **CI-enforced dependency
contract** (`endpoint-import-smoke`): every module must import against the lean
endpoint manifest, and no heavy dep may be reachable from that surface.
`ray[tune]`/`optuna` are heavy and HPO never runs in an endpoint — so HPO code
**cannot** live under `endpoints`.

New surface: **`workbench.training`** — the heavy counterpart (torch/chemprop/
ray/optuna; no lean constraint). It gets **no parallel import-smoke**: the training
image is heavy and we control it, so there's no lean manifest to enforce, and a
smoke would have to install the full stack just to import. The two guards that
actually matter are cheaper: the endpoint smoke's forbidden-list (armed by the
`training` extra) catches any leak of training deps into `endpoints`, and a static
template-import lint enforces the deferred-import invariant below.

**Invariant:** the endpoint imports the whole model script at load time and calls
`model_fn`/`predict_fn`; it never runs `__main__`. Therefore **top-level imports
must be endpoint-safe, and every `workbench.training.*` import must be deferred
inside the template's `__main__`.** The chemprop template already follows this
pattern (its training-only `fit_regression_uq`/`save_regression_uq` imports are
deferred inside `__main__`).

### Staged plan (multi-step)

**Track A — stand up the `workbench.training` surface (DONE):**

1. ✅ Stood up `workbench.training` with its contract docstring (deferred-import
   invariant spelled out).
2. ✅ Added `ray[tune]`/`optuna` as the pyproject **`training` extra** → the
   existing endpoint smoke auto-adds them to its forbidden list, catching any
   leak into `endpoints` for free.
3. ✅ Moved `training_harness` → `workbench.training.training_harness`;
   canonicalized both copy sites (`script_generation`, `features_to_model`),
   which also fixed a latent bug (the `features_to_model` path pointed at a
   non-existent `model_script_utils/`).
4. ✅ Added the template-import lint (`ci/template_import_check.py`, tox env
   `template-import-check`, wired into the Python-lint workflow): a template may
   import `workbench.training.*` only inside `__main__`. Replaces the rejected
   parallel training-smoke.
5. ✅ **Left `uq_harness` in `endpoints`** — verified dual-use: imported at
   top-level by all three deployed templates (xgb/chemprop/pytorch) for the
   endpoint UQ *apply* path. Moving it whole would break every endpoint.

**Track B — the HPO harness (next):**

6. Add `workbench.training.chemprop_hpo` (Ray-Tune/Optuna harness + OOD
   objective), imported **inside `__main__`** of the chemprop template.
7. Add ray/optuna to the `pytorch_chem` training image requirements; make
   `FeaturesToModel` pick a multi-GPU instance (+ longer runtime) when `hpo` is set.
8. Wire the `hpo` block through the template's phase-1/phase-2 flow.

The rule for any future move: **"training-only" = not on any endpoint import
surface** (no top-level import in any deployed template). Dual-use modules stay
lean in `endpoints`.

## Scope

**Chemprop-first.** It's the framework with a native search ecosystem and the PXR
motivation. The `hpo` block + `workbench.training` surface are designed so
XGBoost/PyTorch can slot in later (each needs its own objective/search harness),
but they are **out of scope for the first pass**.

## Defaults (chosen)

Grounded in chemprop's own `basic`/`learning_rate` search groups and our
ensemble-tuned template defaults (`depth=6`, `hidden_dim=700`, `dropout=0.1`,
`ffn_hidden_dim=2000`).

### Search space

Default group = `basic`, adapted (dropout narrowed because the 5-fold ensemble
already regularizes; `ffn_hidden_dim` a categorical that encodes width *and* the
tapered heads — the reason we own the loop — folding in `ffn_num_layers`):

| knob | default search | dist |
|---|---|---|
| `depth` | 2–6 | int, step 1 |
| `hidden_dim` | 300–2400 | int, step 100 |
| `dropout` | 0.0–0.3 | float, step 0.05 |
| `ffn_hidden_dim` | `2000`, `1000`, `500`, `[1024,256,64]`, `[512,128]` | categorical |

Opt-in `learning_rate` group (`search_space="basic+lr"`): `max_lr` 1e-4–5e-3 log ·
`warmup_epochs` 2–10 step 2 · `init_lr`/`final_lr` as ratios of `max_lr`. Off by
default to keep the search 4-D and well-conditioned.

**Never searched** (fixed at configured values): `uq_version` (v1), `max_epochs`,
`patience`, `batch_size`, `split_strategy`, `criterion`, `seed`, foundation settings.

### Budget / compute

- `n_trials = 40` (TPE + ASHA; ASHA kills weak trials early).
- Instance `ml.g6.12xlarge` (4× L4/24GB) — multi-GPU sibling of the current
  chemprop `ml.g6.2xlarge`. Overridable (`g6.24/48xlarge` for bigger sweeps).
- `max_parallel = num_gpus` (→ 4, one trial/GPU). Fractional-GPU (2/GPU) is a
  documented lever, not the default.
- `max_runtime = 24h` (matches the Batch job-def ceiling).

### Objective (prefer OOD, fall back to CV)

- `validation_ids` designated (rows marked `validation`, split out via
  `split_validation_set`) → objective = **MAE on those rows** (`holdout_mae`) —
  the honest OOD read the chemprop template already emits.
- None designated (`validation` frame empty) → **mean 5-fold CV MAE** (`cv_mae`),
  using the existing `n_folds`. No extra user setup.
- Minimized either way; the active objective is **logged** so it's never ambiguous.

## Baseline this doesn't need to beat

The hand-picked "deliberate variant" pattern
(`open_admet_pxr/phase1/pxr_chemprop_tuned_phase1.py`) remains the pragmatic
choice for a small knob space. Automated HPO is for wider searches, not casual
tuning.
