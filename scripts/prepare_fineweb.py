from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.data import sha256_file, write_json
from shared.train import load_config


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    spec, target = config["fineweb"], ROOT / config["fineweb"]["train_tokens"]
    val_target = ROOT / config["fineweb"]["val_tokens"]
    train_count = int(config["pretrain"]["steps"]) * int(config["pretrain"]["seq_len"]) * int(config["pretrain"]["micro_batch_per_gpu"]) * int(config["pretrain"]["gradient_accumulation"]) * 4
    val_count = int(spec["validation_tokens"])
    train_manifest, val_manifest = ROOT / spec["train_manifest"], ROOT / spec["val_manifest"]
    if target.is_file() and val_target.is_file() and train_manifest.is_file() and val_manifest.is_file():
        if len(np.load(target, mmap_mode="r")) == train_count and len(np.load(val_target, mmap_mode="r")) == val_count:
            return
        raise RuntimeError("Existing FineWeb stream does not match the fixed 327,680,000-token protocol")
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["tokenizer_path"], local_files_only=True, use_fast=True)
    train = np.lib.format.open_memmap(target, mode="w+", dtype=np.uint32, shape=(train_count,))
    valid = np.lib.format.open_memmap(val_target, mode="w+", dtype=np.uint32, shape=(val_count,))
    train_docs = (ROOT / "shared/manifests/fineweb_train_documents.jsonl").open("w", encoding="utf-8")
    val_docs = (ROOT / "shared/manifests/fineweb_val_documents.jsonl").open("w", encoding="utf-8")
    stream = load_dataset(spec["dataset"], name=spec["subset"], split="train", streaming=True).shuffle(seed=int(config["seed"]), buffer_size=10_000)
    train_pos = val_pos = 0
    for row in stream:
        text = str(row.get("text", ""))
        if not text:
            continue
        identifier = str(row.get("id", _digest(text)))
        document_hash = _digest(identifier + "\n" + text)
        tokens = np.asarray(tokenizer.encode(text, add_special_tokens=False) + [int(tokenizer.eos_token_id)], dtype=np.uint32)
        destination, handle, position, limit = (valid, val_docs, val_pos, val_count) if int(document_hash[:8], 16) % 20 == 0 else (train, train_docs, train_pos, train_count)
        if position >= limit:
            continue
        take = min(len(tokens), limit - position)
        destination[position:position + take] = tokens[:take]
        handle.write(json.dumps({"document_id": identifier, "document_sha256": document_hash, "tokens": int(take)}) + "\n")
        if destination is train:
            train_pos += take
        else:
            val_pos += take
        if train_pos == train_count and val_pos == val_count:
            break
    train_docs.close(); val_docs.close(); train.flush(); valid.flush()
    if train_pos != train_count or val_pos != val_count:
        raise RuntimeError(f"FineWeb stream incomplete: train={train_pos}/{train_count}, val={val_pos}/{val_count}")
    for name, stream_path, document_path, tokens in (("train", target, ROOT / "shared/manifests/fineweb_train_documents.jsonl", train_count), ("validation", val_target, ROOT / "shared/manifests/fineweb_val_documents.jsonl", val_count)):
        write_json(ROOT / spec[f"{name if name == 'train' else 'val'}_manifest"], {"dataset": spec["dataset"], "subset": spec["subset"], "seed": int(config["seed"]), "ordering": "stream.shuffle(seed=42, buffer_size=10000)", "split_rule": "sha256(document_id + newline + text) modulo 20", "tokens": tokens, "tokenizer": str(config["tokenizer_path"]), "token_stream": str(stream_path), "token_stream_sha256": sha256_file(stream_path), "documents": str(document_path), "documents_sha256": sha256_file(document_path)})


if __name__ == "__main__":
    main()
