"""One-time migration: re-encode legacy raw AWS tag values as base64.

Before June 2026 (commit 6da354f36) tag-safe values were stored raw; now every
value is base64-encoded on write. This script finds artifacts that still carry
raw (legacy) tag values and rewrites them via upsert_workbench_meta(), which
re-encodes everything. Once all accounts are clean, the legacy fallback in
aws_utils.decode_value() can be removed (strict decode).

Covers FeatureSets, Models, and Endpoints (DataSources/Graphs store their
metadata in the Glue catalog, not AWS tags).

Usage:
    python migrate_legacy_tags.py            # dry-run: report legacy values
    python migrate_legacy_tags.py --apply    # rewrite legacy artifacts
"""

import argparse
import base64

from sagemaker.core.common_utils import list_tags

from workbench.api.meta import Meta
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.artifacts.model_core import ModelCore
from workbench.core.artifacts.endpoint_core import EndpointCore


def is_encoded(value: str) -> bool:
    """Strictly check that a tag value is valid base64-encoded UTF-8"""
    if value == " ":  # V3 empty-value workaround, written by current code
        return True
    try:
        base64.b64decode(value, validate=True).decode("utf-8")
        return True
    except Exception:
        return False


def legacy_keys(artifact) -> list:
    """Return the tag keys on this artifact whose values are not base64-encoded"""
    raw_tags = list_tags(artifact.sm_session, artifact.arn())

    # Stitch chunked values back together before checking
    stitched = {}
    values = {}
    for tag in raw_tags:
        key, value = tag["Key"], tag["Value"]
        if "_chunk_" in key:
            base_key, chunk_num = key.rsplit("_chunk_", 1)
            stitched.setdefault(base_key, {})[int(chunk_num)] = value
        else:
            values[key] = value
    for base_key, chunks in stitched.items():
        values[base_key] = "".join(chunks[i] for i in sorted(chunks))

    return [key for key, value in values.items() if not is_encoded(value)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="rewrite legacy artifacts (default: dry-run)")
    args = parser.parse_args()

    meta = Meta()
    artifacts = (
        [FeatureSetCore(name) for name in meta.feature_sets()["Feature Group"]]
        + [ModelCore(name) for name in meta.models()["Model Group"]]
        + [EndpointCore(name) for name in meta.endpoints()["Name"]]
    )

    clean, migrated, failed = 0, 0, []
    for artifact in artifacts:
        keys = legacy_keys(artifact)
        if not keys:
            clean += 1
            continue

        print(f"{artifact.name}: legacy keys {keys}")
        if not args.apply:
            continue

        # Round-trip through the forgiving decode and the (now base64-everything)
        # encode, then re-check in case anything didn't fit the tag budget
        artifact.upsert_workbench_meta(artifact.workbench_meta())
        still_legacy = legacy_keys(artifact)
        if still_legacy:
            failed.append((artifact.name, still_legacy))
            print(f"  STILL LEGACY after rewrite: {still_legacy}")
        else:
            migrated += 1
            print("  migrated")

    print(f"\n{clean} clean, {migrated} migrated, {len(failed)} failed")
    if failed:
        print("Failed artifacts (likely exceeded the 50-tag/256-char budget on re-encode):")
        for name, keys in failed:
            print(f"  {name}: {keys}")
    if not args.apply and clean < len(artifacts):
        print("Dry-run only — rerun with --apply to rewrite")


if __name__ == "__main__":
    main()
