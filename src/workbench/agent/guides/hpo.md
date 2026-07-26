# Hyperparameter Optimization (HPO)

> hyperparameter search: how to launch one, where the results live, how to read them

HPO is not a separate artifact type. It is `to_model()` with an `hpo` block in
`hyperparameters` — the search runs *inside* the single training job and only the
winning config is published. Trials are ephemeral: they never create Workbench
models or endpoints, so a searched model looks like any other model.

Chemprop, XGBoost, and PyTorch, regression only. An otherwise normal `to_model()` call plus:

```python
hyperparameters={"uq_version": "v1", "hpo": {"n_trials": 60}}
```

For the `hpo` block's keys and their defaults, read `run_hpo` in
`workbench/training/hpo_runner.py`; the searched knobs are per-framework, in that
framework's `*_hpo.py`. Read those rather than guessing (see the `code_search` guide).

Cost differs sharply by framework. A chemprop search is a multi-GPU box for hours, so it
belongs on AWS Batch (see the `batch` guide), not inline. An XGBoost trial is seconds,
which makes it the cheap way to try something.

## Finding the results

```python
from workbench.utils.model_utils import get_hpo_results

results = get_hpo_results(model)     # None if the model was not searched
```

That is the only thing you need — it resolves the training job's artifacts for you.
Do **not** go hunting for the training job, the S3 paths, or the CloudWatch logs to
answer "what did HPO pick"; the logs are for diagnosing a *run*, not for reading
results.

`None` means the model was not hyperparameter-searched, so this doubles as the
"is this an HPO model?" check.

It returns the published config, the values below, and `rerank` / `trials` DataFrames.

Both frames carry a `hyperparameters` column: a JSON object of every searched knob and the
value it actually trained at, so `json.loads` gives you a complete, NaN-free record per row.
The knobs differ by framework — read them off the object rather than hardcoding names.

`trials` also carries a `kind` column: the search trials plus one `baseline` row — the
user's own hyperparameters on the same basis — which is the reference line any plot of the
search needs. See the `plotting` guide for parallel coordinates. A `completed` boolean
marks trials that ran the full ensemble (backend-independent); an incomplete trial with a
`value` was pruned, one without ever scored.

## Reading the numbers (the part that misleads)

**The search does not pick the winner.** It shortlists; a second pass re-scores the
finalists *and the user's own untuned hyperparameters* on fresh trainings, and
whichever wins there is published.

There are two same-basis pairs and they must not be mixed:

- `best_value` vs `baseline_value` — the real margin the publish decision turned on.
  **This is the comparison to quote.** The `rerank` frame shares this basis.
- `search_best_value` vs `search_baseline_value` — how the *search* went; same basis as
  every row in `trials`.

When `rerank_fresh_split` is true the two pairs scored on different fold partitions, so a
number from one is not comparable to the other and can even look better. Never present
`search_best_value` as the model's improvement.

**A baseline win is a legitimate outcome, not a failed run.** If `best_config` is
empty or matches the user's own hyperparameters, the search found nothing that beat
their defaults and the untuned model was published. Report that plainly — it means
HPO cost GPU time and confirmed the existing settings. One exception to check: an empty
`best_config` with a null `baseline_value` means the baseline never scored, so nothing
could be measured against it — the untuned config shipped by default, not by winning.

## Judging the search space

Every knob's spec carries its range and its `default`, so a space is inspectable without
running anything — `space_defaults` and the framework's `*_search_space()`.

Winners clustering at a bound is not on its own a reason to widen it. A knob clipped
*mechanically* — an ensemble member hitting its round ceiling — is different, and shows
up only in the training log, never in the objective.

## What to expect

Temper claims. Measured on our own data: HPO improved in-distribution cross-validation but
**lost to stock defaults** on PXR's held-out analog set. Gains on CV do not imply gains on
a new chemical series. Framework defaults are a strong baseline and often win on small
datasets. Quote the run's own numbers rather than a remembered figure.

By default the objective is `cv_mae`, scored out-of-fold on the training rows. Rows
designated via `validation_ids` are held out of training **and** out of the search, so
they stay an honest benchmark — do not suggest tuning on them without saying what it
costs.
