"""One-time migration: rewrite base64-everything tags into the plain/b64: format.

A prior change (June 2026) base64-encoded *every* tag value on write and used a
guess-on-read decode path. That mangled foreign tags and could silently corrupt
plain values that happened to be valid base64. The current scheme instead stores
tag-safe values as plain text and only base64-encodes (with a "b64:" marker) the
values that aren't tag-safe -- so the read side never has to guess.

This script finds artifacts whose tag values are still in the old markerless-base64
format and rewrites them via upsert_workbench_meta(), which re-encodes in the new
format. Values that are already plain (truly-legacy raw, or foreign aws:/human tags)
or already b64:-marked are left alone.

Detection assumption: a non-aws:, markerless value that cleanly base64-decodes to
UTF-8 was written by the prior all-base64 code. This holds for everything Workbench
wrote; a human-added plain tag that happens to be valid base64 is the only false
positive, and aws:-prefixed tags (which can't be modified anyway) are skipped.

Covers FeatureSets, Models, and Endpoints (DataSources/Graphs store their metadata
in the Glue catalog, not AWS tags).

Cleanup (strict-decode release) -- once this has run on ALL accounts and CloudWatch
shows no more "Found legacy encoded tag" warnings, remove the transitional fallback
(grep: remove-legacy-tag-fallback):
  1. aws_utils.py        -- delete _decode_legacy_b64() and the else: branch in decode_value()
  2. tag_tests.py        -- delete the two *_transitional tests; re-add the strict
                            "TWFu stays TWFu" assertion to test_plain_non_base64_pass_through
  3. this script         -- delete it (one-time, done)

Usage:
    python migrate_legacy_tags.py            # dry-run: report what would change
    python migrate_legacy_tags.py --apply    # rewrite affected artifacts
"""

import argparse
import base64
import json

from sagemaker.core.common_utils import list_tags

from workbench.api.meta import Meta
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.artifacts.model_core import ModelCore
from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.utils.aws_utils import B64_MARKER, aws_throttle


def needs_migration(key: str, value: str) -> bool:
    """True if this value is still in the old markerless-base64 format and must be rewritten"""
    if key.startswith("aws:"):  # reserved AWS tags — can't modify, never ours
        return False
    if value == " ":  # V3 empty-value workaround, written by current code
        return False
    if value.startswith(B64_MARKER):  # already new format
        return False
    # Old format: markerless base64 of valid UTF-8 (anything else is already plain-compatible)
    try:
        base64.b64decode(value, validate=True).decode("utf-8")
        return True
    except Exception:
        return False


def recover_value(value: str):
    """Decode an old markerless-base64 value back to the Python value upsert should re-store"""
    decoded = base64.b64decode(value, validate=True).decode("utf-8")
    try:
        return json.loads(decoded)
    except Exception:
        return decoded


@aws_throttle
def stitched_values(artifact) -> dict:
    """Current tag values for an artifact, with chunked values stitched back together.

    Raw list_tags (not the decoded read path) so we can inspect the on-disk encoding. Throttle-wrapped
    because this runs twice per artifact across ~1000 artifacts; delete/upsert are already decorated.
    """
    raw_tags = list_tags(artifact.sm_session, artifact.arn())
    stitched, values = {}, {}
    for tag in raw_tags:
        key, value = tag["Key"], tag["Value"]
        if "_chunk_" in key:
            base_key, chunk_num = key.rsplit("_chunk_", 1)
            stitched.setdefault(base_key, {})[int(chunk_num)] = value
        else:
            values[key] = value
    for base_key, chunks in stitched.items():
        values[base_key] = "".join(chunks[i] for i in sorted(chunks))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="rewrite affected artifacts (default: dry-run)")
    args = parser.parse_args()

    meta = Meta()
    targets = (
        [(FeatureSetCore, name) for name in meta.feature_sets()["Feature Group"]]
        + [(ModelCore, name) for name in meta.models()["Model Group"]]
        + [(EndpointCore, name) for name in meta.endpoints()["Name"]]
    )

    clean, migrated, failed, errored = 0, 0, [], []
    for cls, name in targets:
        # Per-artifact isolation: a transient AWS/network error (or a malformed artifact) skips just
        # this one instead of aborting the whole run. The migration is idempotent, so a re-run retries it.
        try:
            artifact = cls(name)  # construction itself hits AWS, so it's inside the try
            recovered = {k: recover_value(v) for k, v in stitched_values(artifact).items() if needs_migration(k, v)}
            if not recovered:
                clean += 1
                continue

            print(f"{name}: rewriting {list(recovered.keys())}")
            if not args.apply:
                continue

            # Delete first (clears orphan _chunk_ tags — new plain/marked values need fewer chunks),
            # then re-write the recovered values in the new plain/b64: format.
            for key in recovered:
                artifact.delete_metadata(key)
            artifact.upsert_workbench_meta(recovered)

            still = {k: v for k, v in stitched_values(artifact).items() if needs_migration(k, v)}
            if still:
                failed.append((name, list(still.keys())))
                print(f"  STILL OLD FORMAT after rewrite: {list(still.keys())}")
            else:
                migrated += 1
                print("  migrated")
        except Exception as e:
            errored.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR on {name}: {type(e).__name__} — skipping (safe to re-run)")

    print(f"\n{clean} clean, {migrated} migrated, {len(failed)} failed, {len(errored)} errored")
    if failed:
        print("Failed artifacts (likely exceeded the 50-tag/256-char budget on re-encode):")
        for name, keys in failed:
            print(f"  {name}: {keys}")
    if errored:
        print("Errored artifacts (transient/AWS error — re-run to retry):")
        for name, err in errored:
            print(f"  {name}: {err}")
    if not args.apply and clean < len(targets):
        print("Dry-run only — rerun with --apply to rewrite")


if __name__ == "__main__":
    main()
