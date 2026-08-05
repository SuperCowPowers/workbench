#!/usr/bin/env bash
set -euo pipefail

# Back up endpoint data capture files to a backup prefix in the same bucket:
#
#   endpoints/<endpoint>/data_capture/...  ->  data_capture_backups/<endpoint>/data_capture/...
#
# EndpointCore.delete() preserves data_capture/, so captures outlive their endpoints.
# Run this before scripts/admin/delete_data_captures.py. Copies are server-side and
# sync skips objects already present, so an interrupted run resumes where it left off.
# DRY RUN by default: it prints what would be copied. Pass --execute to really copy.
#
#   scripts/admin/backup_data_captures.sh <bucket>              # dry run (default)
#   scripts/admin/backup_data_captures.sh <bucket> --execute    # really copy
#
# Bucket is the Workbench artifacts bucket, e.g. idb-prod-sageworks-artifacts.

BUCKET="${1:?usage: $0 <bucket> [--execute]}"

DRY_RUN=true
[ "${2:-}" = "--execute" ] && DRY_RUN=false

$DRY_RUN && echo "=== DRY RUN (no copies) -- pass --execute to copy ===" \
         || echo "=== COPYING captures to data_capture_backups/ ==="

aws s3 sync "s3://${BUCKET}/endpoints/" "s3://${BUCKET}/data_capture_backups/" \
    --exclude "*" --include "*/data_capture/*" \
    $($DRY_RUN && echo "--dryrun")
