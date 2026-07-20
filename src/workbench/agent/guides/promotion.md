# Model Promotion

> how a challenger becomes champion: ranking, freezing, dated names, retirement

Promotion is a **pipeline step** (`workbench:models/model_promotion.py`) that
picks the best challenger for an endpoint and, if it beats the incumbent,
freezes it and deploys it. It is what produces dated champion models and the
contest reports.

## The sequence

1. **Score challengers.** Challengers are the `-dt` (dynamic training) models
   named in the pipeline step. Each is scored from its `full_cross_fold`
   inference metrics.
2. **Rank.** Classifier → highest `f1`. Regressor → lowest `rmse`.
3. **Compare to the incumbent** — the model currently serving the endpoint
   (`end.get_input()`). No incumbent means the winner is promoted automatically.
4. **Freeze.** `winner.copy(dated_name, owner="Pro")` — a copy, not a retrain.
   The name drops `-dt` and gains a date: `my-model-dt` → `my-model-260715`.
5. **Deploy** the frozen copy onto the champion endpoint, then run
   `test_inference()` and `cross_fold_inference()` on it.
6. **Retire** the dethroned model — it is **deleted**.
7. **Publish** the contest report to `/contests/<endpoint>`, whether or not the
   champion changed.

## Beating the incumbent

- **Classifier:** `f1` must be higher.
- **Regressor:** **both** `rmse` and `mae` must improve. A challenger with
  better RMSE but worse MAE does *not* get promoted.

This is the most common reason a visibly "better" model didn't take the
endpoint — check both metrics before calling it a bug.

## Why a model might never get promoted

- **No `full_cross_fold` metrics.** Challengers without them are skipped with a
  warning, not an error. This is the usual culprit: the model exists, looks
  fine, and is silently ineligible. Check `model.list_inference_runs()`.
- It lost the ranking to a sibling challenger.
- It won the ranking but didn't beat the incumbent on *all* required metrics.

## Retention is current-only

The dethroned model is **deleted**, not archived. Only the current champion and
the live challengers survive, so a promoted model's predecessor is gone. If a
user wants history, that's the contest reports — those persist.

Never run a promotion script casually to "see what happens": it deploys to a
live endpoint and deletes a model.

## Reading the result

```python
from workbench.utils.contest_utils import find_contests
find_contests("pxr-reg-chemprop-mt-logd-260715")
# [{'contest': '/contests/pxr-reg-v1', 'role': 'champion', 'endpoint': 'pxr-reg-v1'}]
```

The dated suffix tells you a model is a promoted champion, and the base name is
what its pipeline declares — see `pipelines` and `contests`.

## Custom policy

The default arbiter is deliberately simple: no thresholds, no notifications, no
config. Clients override it with a `plugin:` script in their pipeline step when
they need different promotion rules.
