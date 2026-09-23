from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.kimi_block.train_pretrain import run as run_kimi_block
from baseline.kimi_full.train_pretrain import run as run_kimi_full
from baseline.native.train_pretrain import run as run_native
from shared.train import load_config
from TA_IBAR_main.train_pretrain import run as run_main


def main() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    config = copy.deepcopy(config)
    config["pretrain"].update({"steps": 3, "warmup_steps": 1, "validation_interval_steps": 3, "checkpoint_interval_steps": 3})
    runs = (("native", run_native, False), ("kimi_full", run_kimi_full, True), ("kimi_block", run_kimi_block, True), ("main_full", run_main, True))
    for method, run, has_routing in runs:
        output = ROOT / "outputs/smoke" / method
        run(config, output)
        if not (output / "final.pt").is_file():
            raise RuntimeError(f"Smoke checkpoint missing for {method}")
        if not has_routing:
            continue
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        if not any(float(row["query_grad_norm"] or 0.0) > 0.0 for row in rows):
            raise RuntimeError(f"Smoke Query gradient is zero for {method}")
        state = torch.load(output / "final.pt", map_location="cpu", weights_only=False)["model"]
        queries = [value.float() for name, value in state.items() if "pseudo_query" in name]
        norms = [value.float() for name, value in state.items() if "key_norm.weight" in name]
        if not queries or not any(torch.count_nonzero(value).item() for value in queries):
            raise RuntimeError(f"Smoke Query did not update for {method}")
        if not norms or not any(not torch.equal(value, torch.ones_like(value)) for value in norms):
            raise RuntimeError(f"Smoke RMSNorm did not update for {method}")
    if int(__import__("os").environ.get("RANK", "0")) == 0:
        (ROOT / "outputs/smoke/PASS").write_text("integrated 4xL4 smoke passed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
