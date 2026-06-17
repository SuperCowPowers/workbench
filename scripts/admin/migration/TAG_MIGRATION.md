# Tag Encoding Migration

Migration from "base64-everything" tag values to the current **plain-when-safe / `b64:`-marked** scheme.

## Why

The old scheme base64-encoded *every* tag value on write and *guessed* on read (`try b64decode, else
plain`). That guess can't tell a value Workbench encoded from a plain value that merely looks like base64,
so it produced two failures:

- **False warnings forever** — every foreign tag (`aws:*`, human-added) failed the decode and logged a
  "legacy" warning. With new foreign tags always arriving, it could never reach zero.
- **Silent corruption** — a plain value that happened to be valid base64 (e.g. `TWFu`) was silently
  decoded to garbage (`Man`), no warning.

## Current scheme

- **Write** (`dict_to_aws_tags`): store the value **plain** if it's tag-safe; otherwise base64-encode it
  and prepend the **`b64:`** marker. Tag-safe = `[A-Za-z0-9 _.:/=+@-]` (what AWS tag values allow).
- **Read** (`decode_value`): `b64:`-marked → decode; everything else passes through untouched. No guessing.
- The marker sits at the front of the value, so it survives chunking (chunks reassemble in order).

## Interim compatibility (the transitional fallback)

`_decode_legacy_b64` in `aws_utils.py` lets **new code read old un-migrated tags** during rollout:

| Reader | Old markerless-b64 tag | New `b64:` / plain tag |
|--------|------------------------|------------------------|
| **New code** (has fallback) | ✓ decoded (warns) | ✓ |
| **Old code** (prior version) | ✓ | ✗ mangles `b64:` values |

Key asymmetry: the fallback makes **new code tolerant of old data**, but **cannot** make old code tolerant
of new data. So updated readers "just work" against any format — but a not-yet-updated reader that hits a
new-format write still breaks. **Update readers before the migration creates new-format tags.**

Transitional cost: the fallback re-introduces the base64 guess for *old* data, so a plain value that looks
like valid base64 (`TWFu`→`Man`) can be mis-decoded during the window. It now logs
`Found legacy encoded tag` so it's visible, and it goes away with the strict cleanup below.

## Rollout checklist

Readers must understand the new format **before** the migration flips anything.

1. [ ] Bump + release workbench to PyPI (with the fallback)
2. [ ] Rebuild images — training / inference / ml_pipelines — pinned to that exact version
3. [ ] Redeploy running endpoints to pick up the new image *(confirm whether endpoints read meta at
       runtime — if they only write tags, this is lower-urgency)*
4. [ ] Tell everyone to update workbench on laptops
5. [ ] **Staging first:** `python migrate_legacy_tags.py` (dry-run) → `--apply` on a non-prod account
6. [ ] **Prod:** `python migrate_legacy_tags.py --apply` per account — *after* steps 1–4 are everywhere

### Verify

7. [ ] Re-run dry-run → reports `N clean, 0 migrated, 0 failed` (idempotent)
8. [ ] Watch CloudWatch for `Found legacy encoded tag` → should trend to zero

## Strict-decode cleanup (later, separate release)

Trigger: migration has run on **all** accounts and the `Found legacy encoded tag` warning is quiet.
`grep remove-legacy-tag-fallback` for the spots.

1. [ ] `aws_utils.py` — delete `_decode_legacy_b64()` and the `else:` branch in `decode_value()`
2. [ ] `tag_tests.py` — delete the two `*_transitional` tests; re-add the strict `TWFu`→`TWFu`
       assertion to `test_plain_non_base64_pass_through`
3. [ ] Delete `migrate_legacy_tags.py` (one-time, done)
4. [ ] Delete this file

## Notes / non-issues

- **DataSources / Graphs** need no migration — their metadata lives in Glue table Parameters (stored
  plain/JSON, never base64-marked), and strict decode reads that fine. Migration covers FeatureSets,
  Models, and Endpoints only.
- **Over-encoding is fixed automatically** — the migration recovers each old value and re-stores it
  through the new encoder, so values that were needlessly base64'd come back as plain.
- **Orphan chunks** — new (shorter) values often need fewer chunks, so the migration `delete_metadata(key)`
  first (clears a key + its `_chunk_` tags) before re-writing.
