#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p logs
nohup bash scripts/bootstrap_then_formal.sh > logs/formal_pipeline.log 2>&1 &
pid=$!
printf '%s\n' "$pid" > logs/formal_pipeline.pid
printf 'PID=%s\nLOG=%s\n' "$pid" "$project_root/logs/formal_pipeline.log"
