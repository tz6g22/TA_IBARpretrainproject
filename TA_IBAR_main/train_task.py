from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.train import load_config, load_weights, train
from .conversion import full_to_task_block
from .modeling import build_full_model, optimizer_factory, routing_metrics

ROOT = Path(__file__).resolve().parents[1]


def run(cfg: dict, task: str, source: Path, partition_path: Path, output: Path) -> Path:
    partition = json.loads(partition_path.read_text(encoding="utf-8"))["block_sizes"]
    def factory(tokenizer):
        full = build_full_model(tokenizer, cfg["model"])
        load_weights(full, source)
        return full_to_task_block(full, partition)
    return train(method_label="ta_ibar_main_block", cfg=cfg, stage=task, output=output,
                 model_factory=factory, optimizer_factory=optimizer_factory, metric_factory=routing_metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--task", choices=("math", "multihop"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.task, args.source_checkpoint, args.partition, args.output)


if __name__ == "__main__":
    main()
