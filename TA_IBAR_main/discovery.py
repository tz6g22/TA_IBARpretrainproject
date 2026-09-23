from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from shared.data import collate_task, encode_examples, load_examples, sha256_file, write_json
from shared.train import load_config, load_weights
from .modeling import build_full_model

TAUS = (0.30, 0.35, 0.37, 0.39, 0.40, 0.41, 0.42, 0.43, 0.45, 0.47, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70)


def interval_matrix(layer_cka: np.ndarray) -> np.ndarray:
    layers = layer_cka.shape[0]
    values = np.full((layers, layers), np.nan, dtype=np.float64)
    for start in range(layers):
        values[start, start] = 1.0
        for end in range(start + 1, layers):
            block = layer_cka[start : end + 1, start : end + 1]
            values[start, end] = float(block[np.triu_indices(end - start + 1, k=1)].min())
    return values


def solve_min_similarity(intervals: np.ndarray, tau: float) -> list[int]:
    """Same DP rule as post-training: fewest blocks, then greatest total coherence."""
    layers = intervals.shape[0]
    states: dict[int, tuple[int, float, tuple[int, ...]]] = {0: (0, 0.0, ())}
    for end in range(1, layers + 1):
        best: tuple[int, float, tuple[int, ...]] | None = None
        for start in range(end):
            previous = states.get(start)
            if previous is None:
                continue
            size = end - start
            score = 1.0 if size == 1 else float(intervals[start, end - 1])
            if size > 1 and (not math.isfinite(score) or score < tau):
                continue
            candidate = (previous[0] + 1, previous[1] + score, previous[2] + (size,))
            if best is None or candidate[0] < best[0] or (candidate[0] == best[0] and candidate[1] > best[1] + 1e-12) or (candidate[0] == best[0] and abs(candidate[1] - best[1]) <= 1e-12 and candidate[2] < best[2]):
                best = candidate
        if best is None:
            raise RuntimeError(f"No legal partition at tau={tau}")
        states[end] = best
    return list(states[layers][2])


def _block_rows(partition: list[int], intervals: np.ndarray) -> list[dict]:
    rows, cursor = [], 0
    for block_id, size in enumerate(partition):
        end = cursor + size - 1
        rows.append({"block_id": block_id, "start_0based": cursor, "end_0based": end, "layers_1based": [cursor + 1, end + 1], "size": size, "min_pairwise_cka": 1.0 if size == 1 else float(intervals[cursor, end])})
        cursor = end + 1
    return rows


def _cka(residuals: list[list[torch.Tensor]], device: torch.device) -> np.ndarray:
    layers = len(residuals)
    values = [torch.cat(parts, dim=0).float() for parts in residuals]
    values = [value - value.mean(dim=0, keepdim=True) for value in values]
    norms = []
    for value in values:
        gram = value.to(device).T @ value.to(device)
        norms.append(float(torch.linalg.matrix_norm(gram).item()))
    matrix = np.eye(layers, dtype=np.float64)
    for left in range(layers):
        current = values[left].to(device)
        for right in range(left + 1, layers):
            other = values[right].to(device)
            denominator = norms[left] * norms[right]
            score = 0.0 if denominator <= 1e-12 else float(((current.T @ other).square().sum() / denominator).item())
            matrix[left, right] = score
            matrix[right, left] = score
    return matrix


def run_discovery(*, cfg: dict, task: str, checkpoint: Path, output: Path) -> None:
    if task not in {"math", "multihop"}:
        raise ValueError(task)
    if not torch.cuda.is_available():
        raise RuntimeError("Discovery requires a CUDA GPU")
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_full_model(tokenizer, cfg["model"])
    load_weights(model, checkpoint)
    model.to(device).eval()
    examples = encode_examples(tokenizer, load_examples(Path(cfg["task_data"][task]["discovery"])), seq_len=int(cfg["task_training"]["seq_len"]))
    if len(examples) != 200:
        raise RuntimeError(f"{task} Discovery must contain exactly 200 cases")
    residuals: list[list[torch.Tensor]] = [[] for _ in range(model.config.num_hidden_layers)]
    hooks = []
    for index, layer in enumerate(model.model.layers):
        def capture(_module, inputs, outputs, index=index):
            residuals[index].append((outputs[0] - inputs[0]).detach().float().cpu())
        hooks.append(layer.register_forward_hook(capture))
    try:
        for item in examples:
            batch = collate_task([item], pad_token_id=int(tokenizer.pad_token_id), device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    matrix = _cka(residuals, device)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-6) or not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
        raise RuntimeError("Linear CKA integrity failure")
    intervals = interval_matrix(matrix)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "similarity_matrix.npy", matrix)
    np.save(output / "interval_similarity.npy", intervals)
    results = []
    for tau in TAUS:
        partition = solve_min_similarity(intervals, tau)
        if sum(partition) != model.config.num_hidden_layers or any(size <= 0 for size in partition):
            raise RuntimeError("DP partition coverage failure")
        blocks = _block_rows(partition, intervals)
        payload = {
            "schema": "linear_cka_min_partition_v1", "method": "linear_cka", "interval_reduction": "min", "task": task,
            "tau": tau, "block_sizes": partition, "num_blocks": len(partition), "num_layers": model.config.num_hidden_layers,
            "blocks": blocks, "similarity_matrix_sha256": sha256_file(output / "similarity_matrix.npy"),
        }
        payload["partition_sha256"] = hashlib.sha256(json.dumps(payload["block_sizes"], separators=(",", ":")).encode()).hexdigest()
        destination = output / f"threshold_{tau:g}" / "partition.json"
        write_json(destination, payload)
        scores = [row["min_pairwise_cka"] for row in blocks if row["size"] > 1]
        results.append({"tau": tau, "partition": partition, "N": len(partition), "max_block": max(partition), "singletons": partition.count(1), "mean_min_cka": float(np.mean(scores)) if scores else 1.0, "median_min_cka": float(np.median(scores)) if scores else 1.0, "global_min_cka": float(np.min(scores)) if scores else 1.0})
    write_json(output / "tau_sweep.json", {"task": task, "rows": results})
    write_json(output / "discovery_manifest.json", {
        "task": task, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "cases": len(examples), "metric": "linear_cka", "interval_reduction": "min",
        "matrix_sha256": sha256_file(output / "similarity_matrix.npy"), "taus": list(TAUS),
    })


def main() -> None:
    import argparse

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs/experiment.yaml")
    parser.add_argument("--task", choices=("math", "multihop"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_discovery(cfg=load_config(args.config), task=args.task, checkpoint=args.checkpoint, output=args.output)


if __name__ == "__main__":
    main()
