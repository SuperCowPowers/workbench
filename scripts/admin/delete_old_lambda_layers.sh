#!/usr/bin/env bash
set -euo pipefail

# Delete old/unused Lambda layers (all versions) across regions.
#
# Lambda has no "delete layer" call -- you delete each version, and layers are
# region-scoped. This walks every layer in each region and deletes all its
# versions. DRY RUN by default: it only PRINTS the delete commands so you can
# eyeball the full list. Pass --execute to actually delete.
#
#   scripts/admin/delete_old_lambda_layers.sh                 # dry run (default)
#   scripts/admin/delete_old_lambda_layers.sh --execute       # really delete
#   KEEP_REGEX='foo|bar' scripts/admin/delete_old_lambda_layers.sh --execute
#
# KEEP_REGEX protects layer names matching it (egrep). Defaults to the current
# minimal layer family so a run after publishing won't nuke the new layer.

REGIONS=(us-east-1 us-west-2)
KEEP_REGEX="${KEEP_REGEX:-workbench-lambda-layer}"   # protect the new kebab-case layer

DRY_RUN=true
[ "${1:-}" = "--execute" ] && DRY_RUN=false

$DRY_RUN && echo "=== DRY RUN (no deletions) -- pass --execute to delete ===" \
         || echo "=== EXECUTING deletions ==="
echo "Regions: ${REGIONS[*]}   Protecting names matching: /${KEEP_REGEX}/"
echo

deleted=0
kept=0
for region in "${REGIONS[@]}"; do
  echo "----- $region -----"
  layers=$(aws lambda list-layers --region "$region" --query 'Layers[].LayerName' --output text)
  for layer in $layers; do
    if echo "$layer" | grep -Eq "$KEEP_REGEX"; then
      echo "KEEP  $layer (matches KEEP_REGEX)"
      kept=$((kept + 1))
      continue
    fi
    versions=$(aws lambda list-layer-versions --layer-name "$layer" --region "$region" \
                 --query 'LayerVersions[].Version' --output text)
    for v in $versions; do
      if $DRY_RUN; then
        echo "would delete: aws lambda delete-layer-version --layer-name $layer --version-number $v --region $region"
      else
        aws lambda delete-layer-version --layer-name "$layer" --version-number "$v" --region "$region"
        echo "deleted $layer:$v ($region)"
      fi
      deleted=$((deleted + 1))
    done
  done
  echo
done

$DRY_RUN && echo "DRY RUN: $deleted version(s) would be deleted, $kept layer(s) kept." \
         || echo "DONE: $deleted version(s) deleted, $kept layer(s) kept."
