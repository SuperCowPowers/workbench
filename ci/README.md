# ci/

Scripts that back the repository's GitHub Actions checks. Each one is runnable
locally with the same invocation CI uses, so you can reproduce a failure without
pushing.

| Script | What it does | Enforced by |
| --- | --- | --- |
| [`lock.sh`](lock.sh) | Regenerates every image's `requirements.lock` from its inputs | [`lockfile-drift.yml`](../.github/workflows/lockfile-drift.yml) |
| [`endpoint_import_smoke.py`](endpoint_import_smoke.py) | Checks every `workbench.endpoints` module imports under the lean endpoint dep surface | [`endpoint-import-smoke.yml`](../.github/workflows/endpoint-import-smoke.yml) |
| [`endpoint_smoke_requirements.txt`](endpoint_smoke_requirements.txt) | The lean dep manifest the smoke test installs against (intersection of every deployed endpoint container's requirements) | — |

## `lock.sh` — lockfile regeneration

Compiles a pinned `requirements.lock` for each image/application from its inputs
(`pyproject.toml` plus, where present, a per-image `requirements.in` for
image-only deps like `fastapi`/`uvicorn`). Run it after bumping a dependency in
`pyproject.toml` or changing an image's dep selection, then commit the resulting
lock diff.

```bash
./ci/lock.sh
```

### Resolution behavior

By default the run is **incremental**: `uv pip compile` reads each existing
`requirements.lock` and reuses its pins as preferred-version hints, so a re-run
only changes what a constraint edit (in `pyproject.toml` / `requirements.in`)
actually forces. New upstream releases on PyPI do **not** move the locks — the
locks move when, and only when, you change a dependency constraint.

Escape hatches when you do want movement:

- **Refresh everything to the latest** (the old aggressive behavior — drops the
  preferred-versions hint and forces fresh PyPI metadata):

  ```bash
  UPGRADE=1 ./ci/lock.sh
  ```

- **Bump a single package**, leaving the rest pinned — add to the relevant
  `compile` call in `lock.sh`:

  ```
  --upgrade-package <name>
  ```

### CI gate

[`lockfile-drift.yml`](../.github/workflows/lockfile-drift.yml) runs `./ci/lock.sh`
and fails if `git diff` shows any change, catching a `pyproject.toml` /
`requirements.in` edit that wasn't accompanied by a fresh lock. Because the
default run is incremental, this check stays green as upstream releases happen —
a diff only appears when a dependency constraint actually changed. To fix a
failure: run `./ci/lock.sh` locally and commit the diff.

## `endpoint_import_smoke.py` — endpoint import contract

Enumerates every module under `workbench/endpoints/` and verifies each imports
cleanly against the lean endpoint dep surface
([`endpoint_smoke_requirements.txt`](endpoint_smoke_requirements.txt)) without
pulling in any heavy orchestration/UI/analysis library transitively. A failure
here means an endpoint container would also fail on its first cold-start after
the change deploys.

```bash
pip install -r ci/endpoint_smoke_requirements.txt
pip install --no-deps -e .
python ci/endpoint_import_smoke.py
```

The script is the single source of truth for coverage: adding a new module under
`workbench/endpoints/` automatically extends the check — no workflow edit needed.
