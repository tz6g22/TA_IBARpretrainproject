"""One-time migration of verified Post-training task rows into this project.

The resulting JSONL files contain prompt/target text and no runtime imports from
the old repository are needed after this script has completed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posttrain-root", type=Path, default=ROOT.parent / "MoiraiBlockPost_train")
    args = parser.parse_args()
    source = args.posttrain_root.resolve()
    if not (source / "src/data/format_tasks.py").is_file():
        raise FileNotFoundError(source)
    os.chdir(source)
    sys.path.insert(0, str(source))
    from src.data.format_tasks import format_task_prompt, format_task_target, load_dataset_pool, load_manifest

    data_config = yaml.safe_load((source / "configs/data_qwen3_1.7b.yaml").read_text(encoding="utf-8"))
    split_rows = load_manifest(source / "outputs/formal_retrain/shared/data/splits.json")
    pools: dict[tuple[str, str], tuple[object, dict]] = {}

    def materialize(record: dict) -> dict:
        dataset, split = str(record["dataset"]), str(record["official_split"])
        key = (dataset, split)
        if key not in pools:
            pool = load_dataset_pool(data_config, dataset)
            pools[key] = pool[split]
        rows, mapping = pools[key]
        row = rows[int(record["row_index"])]
        task = str(record["task"])
        return {"stable_id": str(record["stable_id"]), "content_sha256": str(record["content_sha256"]), "source": dataset, "prompt": format_task_prompt(task, row, mapping), "target": format_task_target(task, row, mapping)}

    task_specs = {
        "math_train.jsonl": ("math", "stage3_adapter_train", {"gsm8k": 1000}),
        "multihop_train.jsonl": ("multihop", "stage3_adapter_train", {"clutrr": 1000}),
        "math_discovery.jsonl": ("math", "stage2_discovery", {"gsm8k": 50, "svamp": 50, "math": 50, "openmathinstruct2": 50}),
        "multihop_discovery.jsonl": ("multihop", "stage2_discovery", {"clutrr": 50, "musique": 50, "hotpotqa": 50, "2wikimultihopqa": 50}),
    }
    manifest: dict[str, dict] = {}
    for name, (task, stage, expected) in task_specs.items():
        selected = sorted((record for record in split_rows if record.get("task") == task and record.get("assigned_split") == stage and record.get("dataset") in expected), key=lambda record: str(record["split_key"]))
        counts = {dataset: sum(record["dataset"] == dataset for record in selected) for dataset in expected}
        if counts != expected:
            raise RuntimeError(f"Unexpected {task}/{stage} composition: {counts}, expected {expected}")
        rows = [materialize(record) for record in selected]
        _write_jsonl(ROOT / "shared/data" / name, rows)
        manifest[name] = {"count": len(rows), "sources": counts, "stable_ids": [row["stable_id"] for row in rows]}
    validation_specs = {
        "math_validation.jsonl": source / "outputs/formal_retrain/shared/math_validation_manifest.json",
        "multihop_validation.jsonl": source / "outputs/formal_1.7b_multihop/shared/multihop_validation_manifest.json",
    }
    for name, path in validation_specs.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [materialize(record) for record in payload["records"]]
        if len(rows) != 32:
            raise RuntimeError(f"Expected 32 validation rows in {path}")
        _write_jsonl(ROOT / "shared/data" / name, rows)
        manifest[name] = {
            "count": len(rows),
            "stable_ids": [row["stable_id"] for row in rows],
            "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (ROOT / "shared/manifests/task_data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
