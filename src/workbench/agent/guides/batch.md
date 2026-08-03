# AWS Batch

> launch heavy or large-scale work onto AWS Batch instead of running it inline

Any stage of the pipeline can be too heavy for the REPL — not just model
training. Building a DataSource or FeatureSet over a large set, an HPO sweep,
live inference over tens of thousands of rows, or a whole multi-step sub-pipeline
run end to end. Run those on **AWS Batch**: the same code, executed in the
Workbench image at scale, off the interactive session.

## Creating an artifact

**Every DataSource, FeatureSet, Model, and Endpoint is the user's call.** Name
what you intend to create and wait for a yes in their next message. Not
conditional on size, speed, or cost — a 200-row FeatureSet needs the same yes as
a chemprop ensemble, and approving an experiment is not approval to build its
artifacts.

Then **default to Batch**. Nearly all artifact creation belongs there. Inline is
the exception and needs a reason you can say out loud: no neural training, no
descriptor or conformer generation, small data you actually checked
(`ds.num_rows()`, `fs.num_rows()`, `len(df)`), finishing in seconds.

Reads create nothing — metrics pulls, plots, `cross_fold_inference()`, inspecting
artifacts, inference on a handful of compounds. Run those inline, no ask.

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

job = launch_batch(code, name="pxr_reg_sweep")   # {"name": "pxr_reg_sweep", ...}
```

`job["name"]` is the stem, which is what `batch_jobs()` matches on — the full job
name (`workbench_pxr_reg_sweep_<timestamp>`) is stamped downstream at submit time.

Launching also starts a watcher that polls every five minutes and reports the
outcome. You'll see it as a `[Batch update: <job> SUCCEEDED]` line at the top of a
later turn — **lead with it**, since the user may not have been at the terminal,
then go look at what the job produced.

A training script must build the **whole chain** — model, endpoint, and both
inference runs — because the job is headless. A `to_model()` that stops there
leaves a model with no endpoint and no metrics (see `making_models`).

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

batch_jobs()              # last 48 hours: name, status, created, runtime, reason
batch_jobs("mppb_reg")    # filter to the one you launched, by the name you gave it
```

### The training job underneath

Batch records nothing about what it submitted, so the link to SageMaker is recovered
from the job's own logs:

```python
from workbench.utils.batch_utils import batch_job_training_jobs, training_job_status

names = batch_job_training_jobs("workbench_mppb_reg_20260722_141615")
training_job_status(names[-1])   # the most recent one
```

`batch_job_training_jobs` returns names in submission order — a script that builds
several models trains several times, and plenty of Batch work never trains at all, so
expect anywhere from zero to many. In `training_job_status`, **`waiting=True` is the
one to call out**: the job is queued for AWS capacity on its instance type, burning
wall-clock without training.

To sweep every training job at once rather than starting from a Batch job:

```python
from workbench.utils.batch_utils import running_training_jobs

running_training_jobs()   # DataFrame of everything in progress, waiting jobs first
```

A job launched as `name="mppb_reg"` appears as `workbench_mppb_reg_<timestamp>`.
It takes a few seconds to show up (SQS → Lambda → Batch). All of these views cover
the last 48 hours, not full history. For full logs, **AWS Batch → Jobs** /
**CloudWatch**. The REPL won't block or report completion — poll `batch_jobs()`
or check the console.
