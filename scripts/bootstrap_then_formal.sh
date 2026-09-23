#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python_bin="${PYTHON_BIN:-python}"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/run_smoke.py
bash scripts/publish_github.sh
exec bash scripts/run_baselines_then_main.sh
