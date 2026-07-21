# Contests

> ranked metrics comparison across a model family; which model performs best

A contest is a **champion/challenger comparison for one endpoint**. The champion
is the model currently serving it; challengers are alternatives scored on the
same inference run, with metric deltas against the champion.

Contests are real, stored artifacts — not just a dashboard view. A model **is**
in a contest if it appears as a row in one.

## Open the dashboard

For "show me the contests" or "how is X doing in its contest", just open the
page — it is a card grid built for exactly this comparison:

```python
from workbench.utils.dashboard_utils import open_page
open_page("contests")
```

## Look them up directly

Contest reports live under `/contests/` in the report store:

```python
from workbench.api import Reports

reports = Reports()
locations = [loc for loc in reports.list() if loc.startswith("/contests/")]
df = reports.get("/contests/aqsol-class")
```

Each row is one model in the contest:

| column | meaning |
|---|---|
| `model` | the model name |
| `role` | `champion` or `challenger` |
| `endpoint` | the endpoint being contested |
| `framework`, `created` | model provenance |
| metrics + `Δ` columns | score, and delta vs. the champion |
| `inference_run` | the capture all rows were scored on |
| `contested` | whether the comparison is active |

The `Δ` columns are the point: a positive delta means the challenger beat the
champion on that metric.

## Which model is best? Use the contest report

For "which pxr model is best" / "compare these models" / "who's the top
performer", the contest report **is the answer** — it is a pre-computed, ranked
metrics table for the whole family, all scored on the same `inference_run`. Read
it instead of pulling metrics model-by-model (that is slower and easy to get
wrong):

```python
from workbench.utils.contest_utils import find_contests
from workbench.api import Reports

loc = find_contests("pxr-reg-chemprop-mt-logd-260715")[0]["contest"]  # -> "/contests/pxr-reg-v1"
report = Reports().get(loc)                                           # champion first, challengers best-first
report[["model", "role", "rmse", "r2", "spearmanr"]]                  # the ranked comparison
```

Rows are already ordered best-first, so the top challenger (or the champion) is
the best performer on the primary metric (rmse for regressors, f1 for
classifiers). Only models entered in the contest appear — if one of the family
isn't there, say so. For an arbitrary set of models that don't share a contest,
`rank_models(models, inference_run)` from `workbench.utils.model_comparison` does
the same ranking on the fly.

## Is this model in a contest?

Use the utility rather than scanning the report store by hand:

```python
from workbench.utils.contest_utils import find_contests

find_contests("pxr-reg-chemprop-mt-logd-260715")
# [{'contest': '/contests/pxr-reg-v1', 'role': 'champion', 'endpoint': 'pxr-reg-v1'}]
```

If it returns nothing, the model genuinely isn't in a contest — say that plainly
rather than concluding contests don't track membership.

## Champions and promotion

The champion is what the endpoint currently serves. Promotion is how a
challenger becomes the champion, so a contest is the evidence behind that
decision — which is why the `Δ` columns and a shared `inference_run` matter:
all rows must be scored on the same data for the comparison to mean anything.

**Champions are usually dated copies.** Promotion copies the winner to
`<base-name>-YYMMDD` and gives that copy the endpoint, so a champion like
`pxr-reg-chemprop-mt-logd-260715` is the promoted form of the pipeline's
`pxr-reg-chemprop-mt-logd`. Strip the suffix when tracing it back to the
pipeline that produced it (see the `pipelines` guide).
