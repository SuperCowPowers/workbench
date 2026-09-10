# Tutorial: Your First Workbench Model

> walk a new-to-Workbench user through the pipeline, from orientation to a scored model

Use this when a user is new to Workbench and wants a walkthrough (a startup nudge
points them here). They already know ML — **don't teach modeling, teach
Workbench's shape.** Run it as a guided conversation: one step, check in, then the
next. Never dump all four steps at once, and never run a step's code before they
say go.

## The one idea to install first

Lead with the mental-model shift — everything else follows from it:

> Every stage — DataSource, FeatureSet, Model, Endpoint — is a **persistent
> artifact in AWS**, not a step in a script. You don't re-run a pipeline; you pick
> a piece up by name (`Model("aqsol-regression")`), enter at any stage, and rebuild
> only what changed. A FeatureSet built last month feeds a model you train today.

Say it once, plainly, then *show* it in step 1. The chain is
`DataSource → FeatureSet → Model → Endpoint`, and the connector between stages is
a pandas DataFrame.

## Step 1 — Orient: what's already here?

Goal: they see the chain and pick something up by name. Read-only, so it's a safe
first move.

- Show what exists — `summary()`, or `CachedMeta().models(details=True)`.
- Pick one artifact up by name and inspect it: `model = Model(name)` then
  `model.details()`. Point out they didn't replay anything to get it.

Mechanics live in the `exploring` and `api_classes` guides — read them, don't
reinvent the calls here.

## Step 2 — Build: a model, end to end

Goal: one scored model from an existing FeatureSet — the shortest chain, and the
payoff step. **The decisions are theirs** — which FeatureSet, framework, target —
so surface the options and ask; don't pick one silently.

The full chain (mechanics and the inline-vs-Batch fork are in `making_models`):
`fs.to_model()` → `model.to_endpoint()` → `end.test_inference()` →
`end.cross_fold_inference()`. A model isn't done until it has an endpoint and both
inference runs — that's what fills in its metrics. The endpoint is serverless, so
it costs nothing to leave up.

## Step 3 — Score: your own data

Goal: run inference on a few rows and watch predictions come back.
`end.inference(df)` returns the frame with predictions appended; pass a
`capture_name` to land the run on the Model. Detail in `endpoints`.

## Step 4 — Compare: which model is best?

Goal: the contest report — a ranked comparison across a model family, best first.
Detail in `contests`. Skip this step if they only built one model.

## Closing

They now have a scored, deployed model and know how to find it again by name. Point
them at the reference guides for depth, and remind them they can just ask you for
any next step — that is the interface.
