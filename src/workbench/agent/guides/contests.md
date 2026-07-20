# Contests

> champion vs challenger model comparisons; who is winning an endpoint

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
