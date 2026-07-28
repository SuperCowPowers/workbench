# Hyperparameter Optimization (HPO)

> hyperparameter search: how to launch one, where the results live, how to read them

HPO is not a separate artifact type. It is `to_model()` with an `hpo` block in
`hyperparameters` — the search runs *inside* the single training job and only the winning
config is published. Trials are ephemeral, so a searched model looks like any other model.

Chemprop, XGBoost, and PyTorch, regression only:

```python
hyperparameters={"uq_version": "v1", "hpo": {"n_trials": 60}}
```

Three methods on `Model` cover everything:

```python
model.hpo_search_space()   # what this model's framework searches, and each knob untuned
model.hpo_results()        # what its search found — None if the model was not searched
model.hpo_importance()     # which knobs actually moved the objective in that search
```

For the block's other keys, read `run_hpo` in `workbench/training/hpo_runner.py`.

## Setting your own ranges and defaults

Two different mechanisms — reach for the simpler one first.

**A default is just a hyperparameter.** Setting a knob alongside the `hpo` block makes it the
baseline the search must beat, and the value for knobs outside the searched groups:
`hyperparameters={"max_lr": 3e-3, "hpo": {...}}`. No special syntax.

**A range needs a `SearchSpace`** — edit the shipped one and pass the JSON:

```python
from workbench.training.hpo_harness import SearchSpace, IntRange

space = SearchSpace("xgboost")          # dict subclass; [] and del are the whole API
space["max_depth"] = IntRange(4, 8, default=6)
hpo={"search_space": space.to_dict()}
```

What you pass is the **whole** space, not a patch — a one-knob dict searches one knob.

Cost differs sharply. A chemprop search is a multi-GPU box for hours, so it belongs on AWS
Batch (see the `batch` guide). An XGBoost trial is seconds.

## Finding the results

`hpo_results()` is the only thing you need. Do **not** hunt the training job, S3 paths, or
CloudWatch logs to answer "what did HPO pick" — logs diagnose a *run*, they don't hold
results. `None` doubles as the "is this an HPO model?" check.

It returns the published config, the values below, and `rerank` / `trials` DataFrames. To
visualize the search, `hpo_plots.hpo_parallel_coordinates(model)` — see the `plotting` guide.

## Reading the numbers (the part that misleads)

**The search does not pick the winner.** It shortlists; a second pass re-scores the
finalists *and the user's own untuned hyperparameters*, and whichever wins there is
published.

Two same-basis pairs, never to be mixed:

- `best_value` vs `baseline_value` — the margin the publish decision turned on. **Quote
  this one.** The `rerank` frame shares its basis.
- `search_best_value` vs `search_baseline_value` — how the *search* went; the basis of
  every `trials` row.

When `rerank_fresh_split` is true these scored on different fold partitions, so one is not
comparable to the other and can even look better. Never present `search_best_value` as the
model's improvement.

**A baseline win is a legitimate outcome, not a failed run.** An empty `best_config` means
nothing beat the user's own defaults, and the untuned model was published — report that
plainly. If `baseline_value` is null too, the baseline never scored, so nothing was
measured against it; say that instead.

Winners clustering at a bound is not on its own a reason to widen it.

## Which knobs mattered

`hpo_importance()` ranks the searched knobs. It returns `importance` (a share, always sums
to 1) and `effect` (the absolute move as a percent of the objective) — **quote both**. A
high share of a negligible total is noise, so a knob is only worth tuning when both are
high. `best` is meaningless when `effect` is small. Estimates come from an adaptive sampler
over a few dozen trials, so report the ordering as directional, never as an ablation.

## What to expect

Temper claims. Measured on our own data: HPO improved in-distribution cross-validation but
**lost to stock defaults** on PXR's held-out analog set. Gains on CV do not imply gains on a
new chemical series. Quote the run's own numbers, not a remembered figure.

The objective defaults to `cv_mae`, out-of-fold on the training rows. Rows designated via
`validation_ids` stay out of training **and** out of the search, so they remain an honest
benchmark — don't suggest tuning on them without saying what it costs.
