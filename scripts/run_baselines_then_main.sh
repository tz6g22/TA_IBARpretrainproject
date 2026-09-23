#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python_bin="${PYTHON_BIN:-python}"
runner=("$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4)

run_pretrain() {
  local module="$1" output="$2"
  "${runner[@]}" -m "$module" --config configs/experiment.yaml --output "$output"
}

run_task() {
  local module="$1" task="$2" source="$3" output="$4"
  "${runner[@]}" -m "$module" --config configs/experiment.yaml --task "$task" --source-checkpoint "$source" --output "$output"
}

run_pretrain baseline.native.train_pretrain outputs/baseline/native/pretrain_full
run_task baseline.native.train_task math outputs/baseline/native/pretrain_full/final.pt outputs/baseline/native/math
run_task baseline.native.train_task multihop outputs/baseline/native/pretrain_full/final.pt outputs/baseline/native/multihop

run_pretrain baseline.kimi_full.train_pretrain outputs/baseline/kimi_full/pretrain_full
run_task baseline.kimi_full.train_task math outputs/baseline/kimi_full/pretrain_full/final.pt outputs/baseline/kimi_full/math
run_task baseline.kimi_full.train_task multihop outputs/baseline/kimi_full/pretrain_full/final.pt outputs/baseline/kimi_full/multihop

run_pretrain baseline.kimi_block.train_pretrain outputs/baseline/kimi_block/pretrain_full
run_task baseline.kimi_block.train_task math outputs/baseline/kimi_block/pretrain_full/final.pt outputs/baseline/kimi_block/math
run_task baseline.kimi_block.train_task multihop outputs/baseline/kimi_block/pretrain_full/final.pt outputs/baseline/kimi_block/multihop

run_pretrain TA_IBAR_main.train_pretrain outputs/TA_IBAR_main/pretrain_full
"$python_bin" -m TA_IBAR_main.discovery --config configs/experiment.yaml --task math --checkpoint outputs/TA_IBAR_main/pretrain_full/final.pt --output outputs/TA_IBAR_main/discovery/math
"$python_bin" -m TA_IBAR_main.discovery --config configs/experiment.yaml --task multihop --checkpoint outputs/TA_IBAR_main/pretrain_full/final.pt --output outputs/TA_IBAR_main/discovery/multihop
touch outputs/TA_IBAR_main/WAITING_FOR_USER_TAU_SELECTION
printf 'WAITING FOR USER TAU SELECTION\n'
