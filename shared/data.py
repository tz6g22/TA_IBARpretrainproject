from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError(f"Empty data file: {path}")
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Example:
    stable_id: str
    content_sha256: str
    prompt: str
    target: str
    source: str


@dataclass(frozen=True)
class EncodedExample:
    stable_id: str
    input_ids: torch.Tensor
    labels: torch.Tensor


def load_examples(path: Path) -> list[Example]:
    rows = read_jsonl(path)
    examples = [
        Example(
            stable_id=str(row["stable_id"]),
            content_sha256=str(row["content_sha256"]),
            prompt=str(row["prompt"]),
            target=str(row["target"]),
            source=str(row["source"]),
        )
        for row in rows
    ]
    if len({item.stable_id for item in examples}) != len(examples):
        raise RuntimeError(f"Duplicate stable IDs in {path}")
    return examples


def encode_examples(tokenizer, examples: Iterable[Example], *, seq_len: int) -> list[EncodedExample]:
    encoded: list[EncodedExample] = []
    for example in examples:
        prompt = tokenizer.encode(example.prompt, add_special_tokens=False)
        target = tokenizer.encode(example.target, add_special_tokens=False) + [int(tokenizer.eos_token_id)]
        if len(target) < 2:
            continue
        prompt = prompt[: max(0, seq_len - len(target))]
        tokens = prompt + target
        if len(tokens) < 2:
            continue
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        labels[: max(0, len(prompt) - 1)] = -100
        encoded.append(EncodedExample(example.stable_id, input_ids, labels))
    if not encoded:
        raise RuntimeError("No task examples retained after tokenization")
    return encoded


def collate_task(examples: list[EncodedExample], *, pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    width = max(item.input_ids.numel() for item in examples)
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for row, item in enumerate(examples):
        width_i = item.input_ids.numel()
        input_ids[row, :width_i] = item.input_ids
        labels[row, :width_i] = item.labels
        attention_mask[row, :width_i] = 1
    return {key: value.to(device, non_blocking=True) for key, value in {
        "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask
    }.items()}


class TokenStream:
    """Fixed non-overlapping token stream shared across all FineWeb methods."""

    def __init__(self, path: Path, *, seq_len: int) -> None:
        self.tokens = np.load(path, mmap_mode="r")
        self.seq_len = int(seq_len)
        if self.tokens.ndim != 1 or self.tokens.dtype != np.uint32:
            raise RuntimeError("FineWeb stream must be a one-dimensional uint32 .npy array")

    def batch(self, *, sequence_offset: int, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        begin = sequence_offset * self.seq_len
        end = begin + batch_size * self.seq_len
        if end > len(self.tokens):
            raise RuntimeError("FineWeb stream is shorter than the configured training budget")
        values = np.asarray(self.tokens[begin:end], dtype=np.int64).reshape(batch_size, self.seq_len)
        input_ids = torch.from_numpy(values.copy()).to(device, non_blocking=True)
        return {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)}


def sequence_offset(*, step: int, accumulation_index: int, rank: int, world_size: int, micro_batch: int, accumulation: int) -> int:
    return (
        step * world_size * micro_batch * accumulation
        + accumulation_index * world_size * micro_batch
        + rank * micro_batch
    )


def task_examples_for_microbatch(examples: list[EncodedExample], *, step: int, accumulation_index: int, rank: int, world_size: int, micro_batch: int, accumulation: int) -> list[EncodedExample]:
    start = sequence_offset(
        step=step,
        accumulation_index=accumulation_index,
        rank=rank,
        world_size=world_size,
        micro_batch=micro_batch,
        accumulation=accumulation,
    )
    return [examples[(start + offset) % len(examples)] for offset in range(micro_batch)]
