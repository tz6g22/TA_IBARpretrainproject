from __future__ import annotations

import argparse
from pathlib import Path

from shared.train import load_config, train
from .modeling import build_model, optimizer_factory

ROOT = Path(__file__).resolve().parents[2]


def run(cfg: dict, output: Path) -> Path:
    return train(method_label="native", cfg=cfg, stage="pretrain", output=output,
                 model_factory=lambda tokenizer: build_model(tokenizer, cfg["model"]),
                 optimizer_factory=optimizer_factory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
