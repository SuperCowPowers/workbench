# Hyperparameter Optimization (HPO)

> hyperparameter search: how to launch one, where the results live, how to read them

HPO is not a separate artifact type. It is `to_model()` with an `hpo` block in
`hyperparameters` — the search runs *inside* the single training job and only the winning
config is published. Trials are ephemeral, so a searched model looks like any other model.

Chemprop, XGBoost, and PyTorch, regression only:

```python
hyperparameters={"uq_version": "v1", "hpo": {"n_trials": 60}}
```

Two methods on `Model` cover everything:

```python
model.hpo_search_space()   # what this model's framework searches, and each knob untuned
model.hpo_results()        # what its search found — None if the model was not searched
```

For the block's other keys, read `run_hpo` in `workbench/training/hpo_runner.py`.

Cost differs sharply. A chemprop search is a multi-GPU box for hours, so it belongs on AWS
Batch (see the `batch` guide). An XGBoost trial is seconds.

## Finding the results

`hpo_results()` is the only thing you need. Do **not** hunt the training job, S3 paths, or
CloudWatch logs to answer "what did HPO pick" — logs diagnose a *run*, they don't hold
results. `None` doubles as the "is this an HPO model?" check.

It returns the published config, the values below, and `rerank` / `trials` DataFrames. See
the `plotting` guide for parallel coordinates.

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

## What to expect

Temper claims. Measured on our own data: HPO improved in-distribution cross-validation but
**lost to stock defaults** on PXR's held-out analog set. Gains on CV do not imply gains on a
new chemical series. Quote the run's own numbers, not a remembered figure.

The objective defaults to `cv_mae`, out-of-fold on the training rows. Rows designated via
`validation_ids` stay out of training **and** out of the search, so they remain an honest
benchmark — don't suggest tuning on them without saying what it costs.
