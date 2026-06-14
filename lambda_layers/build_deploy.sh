#!/usr/bin/env bash
set -e

# Build (and optionally publish) the workbench Lambda layer.
#
# The layer ships ALL of workbench's *source* plus only the allowlisted
# third-party deps (requirements.txt -> networkx, pandas). Other heavy deps
# (torch/awswrangler/sagemaker/...) are intentionally absent: workbench.__init__
# is import-light and the workbench.lambda_layer subset imports with just the
# bundled deps (enforced by tests/lambda_layer/test_layer_dependencies.py).
# boto3/botocore come from the Lambda runtime.
#
# Source is installed via `pip install --no-deps <workbench>` so the layer also
# carries workbench's dist metadata -- the version banner reports the real
# version instead of "unknown".

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
WORKBENCH_ROOT="$(cd "$SCRIPT_DIR/.." &> /dev/null && pwd)"

AWS_ACCOUNT_ID="507740646243"
REGION_LIST=("us-east-1" "us-west-2")
PYTHON_VERSION="3.12"
LAMBDA_ARCH="manylinux2014_x86_64"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
  echo "Usage: $(basename "$0") [--deploy] [--python-version X.Y]"
  echo "  (default) build the layer zip locally"
  echo "  --deploy             publish to AWS Lambda in: ${REGION_LIST[*]} (requires AWS_PROFILE)"
  echo "  --python-version     target Python (default: $PYTHON_VERSION)"
  exit 1
}

DEPLOY=false
while [ $# -gt 0 ]; do
  case $1 in
    --deploy)         DEPLOY=true ;;
    --python-version) PYTHON_VERSION=$2; shift ;;
    -h|--help)        usage ;;
    *)                echo "Unknown option: $1" && usage ;;
  esac
  shift
done

PYVER_TAG="python${PYTHON_VERSION//./}"   # 3.12 -> python312
BUILD_DIR="$SCRIPT_DIR/build"
PY_DIR="$BUILD_DIR/python"                  # Lambda puts <layer>/python on sys.path
ZIP_PATH="$SCRIPT_DIR/workbench_lambda_layer-${PYVER_TAG}.zip"

if [ "$DEPLOY" = true ]; then
  : "${AWS_PROFILE:?AWS_PROFILE environment variable is not set.}"
fi

# -- build --------------------------------------------------------------------
echo "======================================"
echo -e "${YELLOW}🏗️  Building workbench Lambda layer ($PYVER_TAG)${NC}"
echo "======================================"

rm -rf "$BUILD_DIR" "$ZIP_PATH"
mkdir -p "$PY_DIR"

# All workbench source (+ dist metadata), no third-party deps.
echo "Installing workbench source (--no-deps)..."
pip install --no-deps --target "$PY_DIR" "$WORKBENCH_ROOT"

# Only the allowlisted deps, as Lambda-target wheels.
echo "Installing bundled deps for $PYVER_TAG / $LAMBDA_ARCH..."
pip install \
  --target "$PY_DIR" \
  --platform "$LAMBDA_ARCH" \
  --implementation cp \
  --python-version "$PYTHON_VERSION" \
  --only-binary=:all: \
  --upgrade \
  -r "$SCRIPT_DIR/requirements.txt"

# Trim bytecode/caches.
find "$PY_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$PY_DIR" -type f -name "*.pyc" -delete

( cd "$BUILD_DIR" && zip -qr "$ZIP_PATH" python )
echo -e "${GREEN}✅  Built $(basename "$ZIP_PATH") ($(du -h "$ZIP_PATH" | cut -f1))${NC}"

if [ "$DEPLOY" = false ]; then
  echo "Local build complete. Use --deploy to publish to: ${REGION_LIST[*]}."
  exit 0
fi

# -- publish ------------------------------------------------------------------
echo "======================================"
echo "🚀  Publishing layer to: ${REGION_LIST[*]}"
echo "======================================"

for region in "${REGION_LIST[@]}"; do
  layer_name="workbench_lambda_layer-${region}-${PYVER_TAG}"
  echo "Publishing $layer_name in $region..."

  version=$(aws lambda publish-layer-version \
    --layer-name "$layer_name" \
    --description "workbench source + networkx + pandas (boto3 from runtime)" \
    --zip-file "fileb://$ZIP_PATH" \
    --compatible-runtimes "python${PYTHON_VERSION}" \
    --compatible-architectures x86_64 \
    --region "$region" \
    --profile "$AWS_PROFILE" \
    --query Version --output text)

  # Make the layer version readable by any account, so clients attach it by ARN
  # without us granting per-account permissions.
  aws lambda add-layer-version-permission \
    --layer-name "$layer_name" \
    --version-number "$version" \
    --statement-id public-read \
    --principal '*' \
    --action lambda:GetLayerVersion \
    --region "$region" \
    --profile "$AWS_PROFILE" >/dev/null

  arn="arn:aws:lambda:${region}:${AWS_ACCOUNT_ID}:layer:${layer_name}:${version}"
  echo -e "${GREEN}✅  $arn${NC}"
done

echo "======================================"
echo -e "${GREEN}✅  Publish complete. Update docs/lambda_layer/index.md with the ARNs above.${NC}"
echo "======================================"
