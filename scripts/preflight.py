from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.data import load_examples


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    pretrain = config["pretrain"]
    tokens_per_step = 4 * int(pretrain["micro_batch_per_gpu"]) * int(pretrain["gradient_accumulation"]) * int(pretrain["seq_len"])
    checks = {
        "project_root": str(ROOT), "old_posttrain_runtime_dependency": False,
        "gpus_visible": torch.cuda.device_count(), "cuda_available": torch.cuda.is_available(),
        "ddp": True, "layers": config["model"]["num_hidden_layers"], "hidden": config["model"]["hidden_size"],
        "intermediate": config["model"]["intermediate_size"], "heads": config["model"]["num_attention_heads"], "kv_heads": config["model"]["num_key_value_heads"],
        "fineweb_steps": pretrain["steps"], "tokens_per_step": tokens_per_step, "fineweb_tokens": tokens_per_step * int(pretrain["steps"]),
        "fixed_kimi_block_partition": [4, 4, 4], "main_auto_task_training_after_discovery": False,
        "math_train_count": len(load_examples(ROOT / config["task_data"]["math"]["train"])),
        "math_discovery_count": len(load_examples(ROOT / config["task_data"]["math"]["discovery"])),
        "multihop_train_count": len(load_examples(ROOT / config["task_data"]["multihop"]["train"])),
        "multihop_discovery_count": len(load_examples(ROOT / config["task_data"]["multihop"]["discovery"])),
        "fineweb_stream_ready": (ROOT / config["fineweb"]["train_tokens"]).is_file() and (ROOT / config["fineweb"]["val_tokens"]).is_file(),
    }
    checks["pass"] = bool(checks["cuda_available"] and checks["gpus_visible"] == 4 and checks["fineweb_stream_ready"] and checks["fineweb_tokens"] == 327680000)
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not checks["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
