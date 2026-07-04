# First-Class Validation-Set Support

## Principle (why this exists)

A Workbench model needs an honest, held-out set the **training container can see
during training** — for in-training validation / early-stopping across any model
type, and for HPO, whose search objective scores `holdout_mae` on those rows
inside the training job.

Today the per-model training view **drops** `sample_weight == 0` rows before the
CSV is written, so the training container never sees them. That drop also
overloads `sample_weight`, conflating three distinct concepts. This work
de-overloads them and gives every model type a first-class validation set.

This shipped out of the HPO design (see `hpo_support.md`) but stands on its own.

## The overload we're removing

`sample_weight == 0` currently means all of:

1. **A real framework weight** — xgb forwards it to `fit(sample_weight=...)`.
2. **Exclude** — the view's `WHERE ... > 0` drops the row entirely.
3. **Holdout** — the PXR pattern zero-weights rows, then scores them later via
   `endpoint.inference(holdout_df)`.

One number cannot carry three meanings. We split them into orthogonal axes.

## Design: one weight axis + two role bools

| role | designation | in training CSV? | trains? | scored held-out? |
|---|---|---|---|---|
| **train** | default | yes | yes (at `sample_weight`) | via CV OOF |
| **exclude** (outlier/anomaly) | `exclude = true` | **no** (view drops) | no | no |
| **validation** (holdout) | `validation = true` | yes, marked | no | yes (`holdout_mae`) |

- **`sample_weight`** becomes a pure framework weight, forwarded as-is to each
  model script. Each framework decides what to do with it. It never encodes role.
- **`validation`** and **`exclude`** are orthogonal booleans, both default `false`.
  `exclude` wins if somehow both are set.

`sample_weight == 0` stops meaning anything special.

## The per-id roles table (one join)

Roles and weights are **per-model** (they live in the model's own training view,
not the FeatureSet), and datasets can be large — a 100k-row set with 20% held out
is 20k ids. Inlining ids as `IN (...)` would overrun Athena's statement-size
limit. So we carry them the way weights already are: a **sparse supplemental
table**, joined into the view.

One table per model view, only storing rows that differ from the defaults:

```
_<base_table>___<model_view>_roles
    <id_column>     -- the FeatureSet id
    sample_weight   -- default 1.0
    validation      -- default false
    exclude         -- default false
```

View SQL (replaces the current weights-only join + `WHERE > 0`):

```sql
CREATE OR REPLACE VIEW "<base>___<model_view>" AS
SELECT <feature_cols>,
       COALESCE(r."sample_weight", 1.0)  AS sample_weight,
       COALESCE(r."validation",  false)  AS validation,
       COALESCE(r."exclude",     false)  AS exclude
FROM "<base>" t
LEFT JOIN "<roles_table>" r ON t."<id_column>" = r."<id_column>"
WHERE NOT COALESCE(r."exclude", false)
```

When no roles are supplied at all, the view keeps its simple no-join form
(`1.0 AS sample_weight, false AS validation, false AS exclude`).

## Client API

```python
fs.to_model(
    ...,
    sample_weights={id: w, ...},   # pure framework weight; no role meaning
    validation_ids=[...],          # held-out: kept, marked, scored in-training
    exclude_ids=[...],             # dropped from the view entirely
)
```

`FeaturesToModel` normalizes the three inputs into the single sparse roles
DataFrame and hands it to `create_model_training_view`. Sets should be disjoint;
`exclude` takes precedence.

## The util (home + shape)

`workbench.training.validation` — training-only per the `workbench.training`
contract (imported **only inside a template's `__main__`**, never at top level;
enforced by `ci/template_import_check.py`). It's lightweight pandas, but
training-only by role.

```python
def split_validation_set(df, marker="validation"):
    """Return (train_df, val_df). val_df = marked rows (empty if column
    absent or nothing marked). train_df is everything else."""
```

## Template adoption (all three, same shape)

xgb / pytorch / chemprop, right after load + clean:

```python
train_df, val_df = split_validation_set(all_df)
```

- CV folds / OOF / UQ / SHAP run on **`train_df` only**.
- `val_df` gets one scoring pass at the end → `holdout_mae`, appended to
  `validation_predictions.csv` with a marker column distinguishing held-out rows.
- `sample_weight` is forwarded to each framework as a real weight (xgb already
  does; pytorch/chemprop may adopt later).

It's **purely additive** for models that mark a validation set, and a **no-op**
when nothing is marked (util returns an empty `val_df`), so all three templates
can adopt it immediately without changing behavior for existing models.

**Migration is per-model, no coordinated landing.** A model's script and its
training view are co-created at `to_model` time and co-owned by the model.
Existing models keep their current script + view until they retrain; a retrain
regenerates both together from the current template + view code. Custom scripts
that don't call the util are unaffected — they just see the extra columns.

## First pass vs. later

- **First pass:** eval-only — route validation rows out of train/CV and score
  them held-out. This is the minimal honest-validation capability and unblocks
  HPO's `holdout_mae` objective.
- **Later (per template):** use the validation set as the early-stopping monitor
  (chemprop/pytorch) instead of a CV fold.

## Constraints

- Keep `uq_version` conventions and existing behavior intact for models that
  don't mark a validation set.
- Base off `main` (independent of the in-flight `hpo_support` branch).
- User runs their own commits and AWS mutations; Claude preps + verifies.
</content>
</invoke>
