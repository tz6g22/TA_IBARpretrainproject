from __future__ import annotations

import argparse
from pathlib import Path

from shared.train import load_config, load_weights, train
from .modeling import build_model, optimizer_factory, routing_metrics

ROOT = Path(__file__).resolve().parents[2]


def run(cfg: dict, task: str, source: Path, output: Path) -> Path:
    def factory(tokenizer):
        model = build_model(tokenizer, cfg["model"])
        load_weights(model, source)
        return model
    return train(method_label="kimi_block", cfg=cfg, stage=task, output=output,
                 model_factory=factory, optimizer_factory=optimizer_factory, metric_factory=routing_metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--task", choices=("math", "multihop"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.task, args.source_checkpoint, args.output)


if __name__ == "__main__":
    main()
