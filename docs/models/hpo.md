# Hyperparameter Optimization (HPO)

<figure style="text-align: center;">
  <img src="../images/hpo_overview.svg" alt="HPO wrapped over model creation" style="height: 260px;">
</figure>

HPO wraps normal model creation. The search driver hands a set of hyperparameters to the
same training code that publishes your models, gets a score back, and repeats — all inside
a **single** training job. Trials are ephemeral: they never create Workbench models or
endpoints, so a searched model looks like any other model.

Available for ChemProp, XGBoost, and PyTorch, regression only.

## Running a search

An otherwise normal `to_model()` call plus an `hpo` block:

```python
from workbench.api import FeatureSet, ModelType, ModelFramework

fs = FeatureSet("aqsol_features")

model = fs.to_model(
    name="sol-xgb-hpo",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.XGBOOST,
    target_column="solubility",
    feature_list=fs.feature_columns,
    description="Solubility regression (hyperparameter-searched)",
    tags=["solubility", "hpo"],
    hyperparameters={"uq_version": "v1", "hpo": {"n_trials": 250}},
)
```

The winning configuration becomes the model's hyperparameters, so the published model is
deployed and used exactly like any other.

### The `hpo` block

| key | default | what it does |
|---|---|---|
| `n_trials` | `60` | search budget — how many configurations to try |
| `search_space` | all groups | which knob groups to search — see below |
| `metric` | `cv_mae` | objective: `cv_mae` (out-of-fold) or `holdout_mae` |
| `rerank_top_k` | `5` | finalists re-scored in the second stage (`0` disables it) |
| `backend` | `auto` | `optuna` (serial) or `ray` (parallel, needs a GPU box) |
| `max_parallel` | `1` | concurrent trials (Ray only) |
| `n_folds` | model's `n_folds` | ensemble size per trial — for cheap validation runs only |

The searched knobs differ per framework, and a model knows its own — `hpo_search_space()`
dispatches on the model's framework and returns one row per knob, carrying its range **and**
where it sits untuned. A space is judgeable without running anything:

```python
model.hpo_search_space()           # every knob for this model's framework
```

Three columns are pinned — `knob`, `default`, `dist` — and everything specific to the
distribution rides in a `spec` JSON object, so knobs of different kinds share one table
without leaving holes:

| knob | default | dist | spec |
|---|---|---|---|
| `max_depth` | 7 | int | `{"low": 3, "high": 16, "step": 1}` |
| `learning_rate` | 0.05 | float | `{"low": 0.003, "high": 0.3, "log": true}` |
| `ffn_hidden_dim` | 1800 | choice | `{"options": [300, 600, "1024-256-64", ...]}` |

`json.loads` the `spec` cell to read it. This is always the framework's full space.

To search only part of it, set `hpo["search_space"]` to a group name or a `+`-joined
combination. `basic` is capacity everywhere; the second group differs because the frameworks'
second lever does:

| framework | groups |
|---|---|
| ChemProp, PyTorch | `basic` + `optimizer` (learning rate, batch size) |
| XGBoost | `basic` + `reg` (sampling and penalty terms) |

So `hpo={"search_space": "basic"}` spends the whole budget on architecture — useful when you
already trust your optimizer settings.

## Cost differs sharply by framework

An XGBoost trial is seconds, which makes a 250-trial search the cheap way to try something.
A ChemProp search trains a full ensemble per trial on a GPU — that is a multi-GPU box for
hours, so it belongs on AWS Batch rather than run inline.

## Selection is two-stage

A search reports the *minimum* over many noisy estimates, so its winning value is
optimistically biased — the winner may simply have drawn the luckiest evaluation. Workbench
therefore treats the search as a **shortlist**, not a decision:

1. **Search** — sample the space, and prune configurations that are already off the pace.
2. **Re-rank** — re-score the top finalists *and your own untuned hyperparameters* on fresh
   trainings, then publish whichever wins.

Carrying your own hyperparameters through the second stage is what bounds the downside: a
search that finds nothing real loses to them, and **your original settings get published
unchanged**. Ties go to your settings.

That makes a baseline win a legitimate outcome rather than a failed run — it means the
search spent the compute and confirmed your existing configuration.

## Reading the results

```python
results = model.hpo_results()       # None if the model was not searched
```

That resolves the training job's artifacts for you — no need to hunt for S3 paths or
CloudWatch logs. A `None` return doubles as the "was this model searched?" check.

It returns the published configuration, the values below, and `rerank` / `trials`
DataFrames.

There are two same-basis value pairs, and mixing them is the easy mistake:

| pair | meaning |
|---|---|
| `best_value` vs `baseline_value` | the real margin the publish decision turned on — **the one to quote** |
| `search_best_value` vs `search_baseline_value` | how the search itself went; same basis as the `trials` rows |

When `rerank_fresh_split` is `true` the two pairs scored on different fold partitions, so a
number from one is not comparable to the other and can even look better. Never present
`search_best_value` as the model's improvement.

Both DataFrames carry a `hyperparameters` column: a JSON object of every searched knob and
the value it actually trained at, so each row is a complete, NaN-free record. The `trials`
frame also carries a `kind` column — the trials plus one `baseline` row, which is the
reference line any plot of the search needs.

## How it fits together

<figure style="text-align: center;">
  <img src="../images/hpo_details.svg" alt="HPO component interactions" style="height: 330px;">
</figure>

Everything on the left is per-framework — one model template, one adapter, and one fold
trainer for each of ChemProp, XGBoost, and PyTorch. Everything on the right is shared and
knows nothing about any framework.

A template hands its `hpo` block to its adapter, which calls `run_hpo`. From there control
inverts: the runner drives the search and calls *back* into the adapter for the
framework-specific parts — the search space, how to train and score one candidate, and how a
winning config merges back into the hyperparameters. That boundary is why all three
frameworks produce identical artifacts and `hpo_results()` needs no per-framework logic.

The detail that makes a search result mean anything is at the bottom left: a trial and the
final published model train through the **same** `train_*_fold` function. A configuration is
therefore scored in exactly the regime it ships in.

## What to expect

Temper expectations. Framework defaults are a strong baseline and often win on small
datasets, and the literature finds ChemProp HPO is roughly a coin flip against defaults
there. Measured on our own data, HPO improved in-distribution cross-validation but **lost to
stock defaults** on a held-out analog series — gains on cross-validation do not imply gains
on a new chemical series.

By default the objective is `cv_mae`, scored out-of-fold on the training rows. Rows
designated through `validation_ids` are held out of training **and** out of the search, so
they remain an honest benchmark. Opt into tuning toward them with
`hpo={"metric": "holdout_mae"}` only when that set exists to be tuned against — doing so
makes its own final score optimistic.

---

## Questions?

<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
