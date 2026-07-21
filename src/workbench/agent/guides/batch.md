# AWS Batch

> launch heavy or large-scale work onto AWS Batch instead of running it inline

Some work is too heavy for the REPL — model training, HPO sweeps, cross-fold on a
big set, inference over tens of thousands of rows. Run those on **AWS Batch**: the
same code, executed in the Workbench image at scale, off the interactive session.

## When to launch vs. run inline

Estimate the weight of the work and decide. The honest signal is the operation
plus the data scale — check it at runtime (`fs.num_rows()`, the length of the eval
set) rather than guessing a clock.

**Launch to Batch:**

- Any neural / chemprop training, or an HPO sweep.
- Cross-fold or full inference over a large set (roughly tens of thousands of rows
  or a whole FeatureSet).
- Anything you'd expect to run for many minutes.

**Run inline (REPL):**

- Metrics pulls, plots, inspecting a single model or artifact.
- Inference on a handful of compounds.
- XGBoost or similar on a small set.

The rule is the spirit, not a stopwatch: if the work is heavy or large-scale,
launch it; if it's quick and interactive, keep it inline.

## Confirm first — Batch costs real compute

Unlike serverless endpoints (free when idle), a Batch job is **billable compute**.
Before launching, tell the user what you're about to run — the script's purpose,
the `size` tier, and what artifact it will produce — and wait for a yes. Never
launch a Batch job inside a larger block of code without that confirmation.

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
print("trained", model.name)
'''

job = launch_batch(code, name="pxr_reg_sweep", size="medium")   # small | medium | large
```

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
