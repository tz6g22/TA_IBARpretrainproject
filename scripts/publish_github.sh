#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
command -v gh >/dev/null || { echo 'GitHub CLI is required'; exit 1; }
gh auth status
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create TA_IBARpretrainproject --private --source=. --remote=origin
fi
git add .
git commit -m 'Refactor TA-IBAR train-from-scratch into independent main and baseline implementations' || true
git push -u origin main
