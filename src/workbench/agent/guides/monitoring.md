# Endpoint Monitoring

> endpoint drift monitoring: baselines, schedules, alerts

A `Monitor` wraps a deployed Endpoint with data capture and model-quality
monitoring: capture live traffic, compare it against a baseline built from the
training data, and flag drift.

```python
mon = Endpoint("my-model").monitor()   # or Monitor("my-model")
mon.summary()
mon.details()
```

## Serverless does not support monitoring schedules

**Check this first.** Workbench endpoints are serverless by default, and
SageMaker monitoring schedules don't work on serverless:

```python
end.is_serverless()    # True -> no monitoring schedule
```

On a serverless endpoint, `create_monitoring_schedule()` logs a warning and
**returns without doing anything** — no error, no schedule. `create_baseline()`
still works but warns that monitoring won't run.

So if a user wants scheduled monitoring, the endpoint has to be **realtime**
(`serverless=False`), which is persistent billed compute — confirm with them
before rebuilding it that way. Don't quietly set up a monitor that will never
run.

## Setup

Three one-time steps, in order:

```python
mon.enable_data_capture(capture_percentage=100)   # start capturing traffic
mon.create_baseline()                             # from the FeatureSet training data
mon.create_monitoring_schedule(schedule="hourly") # or "daily"
```

Order matters — `create_monitoring_schedule()` bails with a warning if no
baseline exists. `create_baseline()` is a no-op when one is already there unless
you pass `recreate=True`, and it writes three files: `baseline.csv`,
`constraints.json`, `statistics.json`.

Data capture is also what feeds `capture_name` inference runs, so it is worth
enabling even on endpoints that can't be scheduled.

## Reading results

```python
mon.get_baseline()             # the reference DataFrame
mon.get_statistics()           # per-column stats from the baseline
mon.get_constraints()          # the rules violations are judged against
mon.get_monitoring_results()   # most recent monitoring runs
```

A violation means live traffic drifted from the baseline distribution — new
chemistry, a changed upstream pipeline, or a genuine data problem. It does not
by itself mean the model got worse; check predictions before concluding that.

If a constraint is too strict for a column that is legitimately variable,
`update_constraints()` adjusts it rather than disabling monitoring.

## Housekeeping

```python
mon.baseline_exists()
mon.monitoring_schedule_exists()
mon.delete_monitoring_schedule()
mon.setup_alerts(notification_email, threshold=1)
```

`setup_alerts()` sends email on violations — an outward-facing action, so
confirm the address with the user before calling it.

## Drift vs. applicability domain

Monitoring answers "is incoming data different from training data?" at the
*population* level. For "should I trust this one prediction?", that's the UQ
confidence and applicability-domain story — see the `uq` guide. They are
complementary: drift explains *why* confidence might be dropping.
