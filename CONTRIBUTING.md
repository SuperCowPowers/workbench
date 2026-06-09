## Contributing to Workbench

Thank you for your interest in contributing to Workbench! 

All contributions will fall under the existing project [license](https://github.com/SuperCowPowers/workbench/blob/main/LICENSE). As part of our PR Process we have an auto checklist that the contributor should try to check off and a Contributor License Agreement (CLA) will pop up (just once) if you haven't contributed before.

You can contribute by reporting bugs, suggesting features, or submitting code changes. Feel free to browse open issues or propose your own changes.
If you have any questions or need assistance, don't hesitate to reach out to us at workbench@supercowpowers.com.

### Automated & AI-assisted contributions

The top issue here is volume—please don't spam the repo with a big batch of PRs. Just keep it to 3 or 4 open at a time; once those get reviewed/approved/merged, send a few more.

- **A PR should link to an open issue.** PRs without a linked issue may be closed without a full review.
- **Please disclose if a PR was largely automated or AI-generated.**
- **Make sure it runs.** Confirm the test suite passes locally before opening the PR; a change you haven't been able to run isn't ready for review.

### Local development environment (uv)

Workbench uses [uv](https://docs.astral.sh/uv/) to manage its virtualenv and lockfile. uv creates the environment at `./.venv` **inside the repo** (note: older pyenv setups kept it outside). For day-to-day work, activate it once and run tools directly — uv only re-resolves when you explicitly run `uv sync` / `uv lock`:

```bash
uv sync                 # build/update ./.venv from pyproject.toml + uv.lock
source .venv/bin/activate
pytest tests/...        # then python / pytest / black / workbench run plainly
```

#### Running tox under uv

tox is **not** on your PATH after a uv migration. Run it one of two ways:

```bash
uvx tox -e lint         # ephemeral, no install
# or, to restore the bare `tox` command permanently:
uv tool install tox
tox -e lint
```

**Do not use `uv run tox`** — it resolves the entire project env first and can choke on heavy transitive deps. Keep tox isolated from the project via `uvx` or `uv tool`.

#### Gotchas we've already hit

- **flake8 scanning `.venv`** — because the virtualenv now lives in-repo, linters will scan installed packages unless excluded. We exclude `.venv` in [`.flake8`](.flake8) (`extend-exclude`). If you add a new linter, exclude `.venv` there too.
- **Unbounded transitive deps** — uv does a *universal* resolution (py3.10–3.13, all platforms) and will backtrack to ancient versions of a dependency that upstream declares without a floor. Example: `sagemaker-serve` requires `torch` unconstrained, which resolved to a py313-incompatible `torch 2.0.1`. We pin a floor in `[tool.uv] constraint-dependencies` in [`pyproject.toml`](pyproject.toml) — a constraint bounds a transitive package without declaring it as our own dependency. Remove such pins if/when upstream specifies its own floor.

We look forward to your contributions to Workbench!

<img align="right" src="docs/images/scp.png" width="180">
