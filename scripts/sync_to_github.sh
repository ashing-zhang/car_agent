#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   car_agent GitHub Sync Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}Error: Not inside a git repository.${NC}"
    exit 1
fi

REMOTE_NAME="${1:-origin}"
BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD)"

if ! git remote get-url "$REMOTE_NAME" > /dev/null 2>&1; then
    echo -e "${RED}Error: Remote '$REMOTE_NAME' does not exist.${NC}"
    echo "Available remotes:"
    git remote -v
    exit 1
fi

REMOTE_URL="$(git remote get-url "$REMOTE_NAME")"
echo -e "Remote:    ${YELLOW}$REMOTE_NAME ($REMOTE_URL)${NC}"
echo -e "Branch:    ${YELLOW}$BRANCH_NAME${NC}"
echo ""

STATUS_OUTPUT="$(git status --porcelain)"
if [ -z "$STATUS_OUTPUT" ]; then
    echo -e "${YELLOW}No changes detected. Working tree is clean.${NC}"

    echo ""
    echo -e "${GREEN}Pulling latest changes from remote...${NC}"
    if git pull "$REMOTE_NAME" "$BRANCH_NAME"; then
        echo -e "${GREEN}Repository is already up to date with $REMOTE_NAME/$BRANCH_NAME.${NC}"
    else
        echo -e "${RED}Error: Failed to pull from remote.${NC}"
        exit 1
    fi
    exit 0
fi

echo "Changes detected:"
echo "$STATUS_OUTPUT"
echo ""

if [ -n "$2" ]; then
    COMMIT_MSG="$2"
else
    read -p "Enter commit message (or press Enter for auto message): " COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Auto sync: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
fi

echo ""
echo -e "${GREEN}Staging all changes...${NC}"
git add -A

echo -e "${GREEN}Committing changes...${NC}"
if git commit -m "$COMMIT_MSG"; then
    echo -e "${GREEN}Commit successful.${NC}"
else
    echo -e "${RED}Error: Commit failed.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Pulling latest changes from remote (to avoid conflicts)...${NC}"
if git pull "$REMOTE_NAME" "$BRANCH_NAME" --rebase; then
    echo -e "${GREEN}Pull successful.${NC}"
else
    echo -e "${RED}Error: Pull failed. Resolve conflicts and run the script again.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Pushing to $REMOTE_NAME/$BRANCH_NAME...${NC}"
if git push "$REMOTE_NAME" "$BRANCH_NAME"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Sync completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Commit: $(git log -1 --oneline)"
else
    echo -e "${RED}Error: Push to remote failed.${NC}"
    exit 1
fi
