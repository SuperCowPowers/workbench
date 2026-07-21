# AWS Batch

> launch heavy or large-scale work onto AWS Batch instead of running it inline

Any stage of the pipeline can be too heavy for the REPL — not just model
training. Building a DataSource or FeatureSet over a large set, an HPO sweep,
live inference over tens of thousands of rows, or a whole multi-step sub-pipeline
run end to end. Run those on **AWS Batch**: the same code, executed in the
Workbench image at scale, off the interactive session.

## When to launch vs. run inline

Estimate the weight of the work and decide. The honest signal is the operation
plus the data scale — check it at runtime (`ds.num_rows()`, `fs.num_rows()`, the
length of the eval set) rather than guessing a clock.

**Launch to Batch:**

- Any neural / chemprop training, or an HPO sweep.
- A heavy `to_features()` build — molecular descriptors, fingerprints, or
  conformers over a large DataSource.
- Live inference (`end.inference(df)`) over a large set — roughly tens of
  thousands of rows or a whole FeatureSet — where the endpoint scores every row.
  (Not `cross_fold_inference()`, which just pulls training-time predictions from
  S3 and stays quick regardless of size — keep that inline.)
- A multi-step sub-pipeline (`ds → fs → model → endpoint`) run end to end, where
  any one stage is heavy or the whole chain is long.
- Anything you'd expect to run for many minutes.

**Run inline (REPL):**

- Metrics pulls, plots, inspecting a single model or artifact.
- A small DataSource / FeatureSet build over a modest set.
- Inference on a handful of compounds.
- XGBoost or similar on a small set.

The rule is the spirit, not a stopwatch: if the work is heavy or large-scale,
launch it; if it's quick and interactive, keep it inline.

## Ephemeral compute — no cost gate

A Batch job is ephemeral: it runs, produces its artifact, and stops — like any
training job. It is **not** persistent compute, so it needs no cost confirmation
the way a realtime endpoint does. Don't gate a launch on billing. Launching is
part of the work you're already planning with the user (the plans-and-decisions
rules in `general`), not a separate billing decision.

A read-only session can't submit (the SQS send is a write); report that plainly
rather than working around it.

## Launching

```python
from workbench.utils.batch_utils import launch_batch

code = '''
from workbench.api import FeatureSet, ModelType
fs = FeatureSet("pxr_features")
model = fs.to_model(name="pxr-reg-chemprop-sweep", model_type=ModelType.REGRESSOR,
                    target_column="pec50", hyperparameters={"uq_version": "v1"})
end = model.to_endpoint(name="pxr-reg-chemprop-sweep", tags=["pxr-reg-chemprop-sweep"])
end.test_inference()
end.cross_fold_inference()
'''

job = launch_batch(code, name="pxr_reg_sweep", size="medium")   # small | medium | large
```

A training script must build the **whole chain** — model, endpoint, and both
inference runs — because the job is headless. A `to_model()` that stops there
leaves a model with no endpoint and no metrics (see `making_models`).

`launch_batch` writes the code to a temp file, uploads it, and submits the job. It
prints the submission log (message id, monitoring locations) and returns
`{"name", "size", "s3_path"}`.

## The script is standalone — not the REPL

The job runs in a **fresh process**, so the code does not see the REPL namespace.
Two consequences:

- Make it self-contained: its own imports, explicit artifact names. Don't
  reference variables from the session.
- Results come back as **Workbench artifacts**, not a return value. A training job
  leaves a new Model; query it afterward with the normal API (`Model(name)`,
  `list_inference_runs()`, or a contest) once the job finishes.

## Monitoring

The launch is asynchronous — it returns immediately, the job runs on its own.
Check status from the REPL with `batch_jobs()`:

```python
from workbench.utils.batch_utils import batch_jobs

batch_jobs()              # recent jobs: name, status, created, runtime, reason
batch_jobs("mppb_reg")    # filter to the one you launched, by the name you gave it
```

A job launched as `name="mppb_reg"` appears as `workbench_mppb_reg_<timestamp>`.
It takes a few seconds to show up (SQS → Lambda → Batch), and terminated jobs are
only retained for a limited window (at least ~24h, often several days) — a recent
view, not full history. For full logs, **AWS Batch → Jobs** / **CloudWatch**. The
REPL won't block or report completion — poll `batch_jobs()` or check the console.
