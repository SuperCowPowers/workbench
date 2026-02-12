#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
for cmd in python tox twine git; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd is not installed${NC}"
        exit 1
    fi
done

# Make sure we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo -e "${RED}Error: Not on main branch (currently on '$BRANCH')${NC}"
    exit 1
fi

# Make sure working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: Working directory is not clean. Commit or stash changes first.${NC}"
    git status --short
    exit 1
fi

# Get version tag
CURRENT=$(git describe --tags --abbrev=0 2>/dev/null || echo "none")

if [ -z "$1" ]; then
    if [ "$CURRENT" = "none" ]; then
        echo -e "${RED}Error: No existing tags found and no version specified.${NC}"
        echo -e "${YELLOW}Usage: ./publish.sh <version>${NC}"
        exit 1
    fi
    # Auto-increment: bump the last number in the version
    # Strip the 'v' prefix, split on '.', increment the last segment
    BASE="${CURRENT#v}"
    PREFIX="${BASE%.*}"
    LAST="${BASE##*.}"
    NEXT=$((LAST + 1))
    VERSION="${PREFIX}.${NEXT}"
    echo -e "Current tag: ${GREEN}${CURRENT}${NC}"
    echo -e "Next version: ${GREEN}v${VERSION}${NC}"
    read -p "Publish v${VERSION}? [y/N] " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Aborted.${NC}"
        exit 0
    fi
else
    VERSION="$1"
    # Ensure the 'v' prefix is stripped from the version number (we add it ourselves)
    VERSION="${VERSION#v}"
fi

echo -e "${GREEN}=== Publishing workbench v${VERSION} ===${NC}"

# Step 1: Run linting and tests
echo -e "\n${YELLOW}[1/6] Running tox (lint + tests)...${NC}"
tox

# Step 2: Clean build artifacts
echo -e "\n${YELLOW}[2/6] Cleaning build artifacts...${NC}"
make clean
rm -rf dist/ build/

# Step 3: Tag the version
echo -e "\n${YELLOW}[3/6] Tagging v${VERSION}...${NC}"
git tag "v${VERSION}"
git push --tags

# Step 4: Build
echo -e "\n${YELLOW}[4/6] Building package...${NC}"
python -m build

# Step 5: Upload to PyPI
echo -e "\n${YELLOW}[5/6] Uploading to PyPI...${NC}"
twine upload dist/* -r pypi

# Step 6: Push
echo -e "\n${YELLOW}[6/6] Pushing to origin...${NC}"
git push

echo -e "\n${GREEN}=== Successfully published workbench v${VERSION} ===${NC}"
echo -e "https://pypi.org/project/workbench/${VERSION}/"
