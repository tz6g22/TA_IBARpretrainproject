from __future__ import annotations

import json
import math
import os
import random
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from .data import (
    TokenStream,
    collate_task,
    encode_examples,
    load_examples,
    read_jsonl,
    sequence_offset,
    sha256_file,
    task_examples_for_microbatch,
    write_json,
)
from .evaluation import task_metrics

ModelFactory = Callable[[object], torch.nn.Module]
OptimizerFactory = Callable[[torch.nn.Module, dict, str], torch.optim.Optimizer]
MetricFactory = Callable[[torch.nn.Module], tuple[float | None, float | None]]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def runtime() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: run this project on the 4xL4 GPU node")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank, "device": torch.device("cuda", local_rank)}


def rank0(ctx: dict) -> bool:
    return ctx["rank"] == 0


def barrier(ctx: dict) -> None:
    if ctx["world_size"] > 1:
        dist.barrier()


def all_reduce_scalar(value: float | int, *, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    item = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(item, op=op)
    return float(item.item())


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(model, ctx: dict):
    model.to(ctx["device"])
    if ctx["world_size"] > 1:
        return DDP(model, device_ids=[ctx["local_rank"]], output_device=ctx["local_rank"], broadcast_buffers=False)
    return model


class CosineSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer, *, total: int, warmup: int, min_lr: float) -> None:
        self.optimizer, self.total, self.warmup, self.min_lr = optimizer, int(total), int(warmup), float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def step(self, position: int) -> None:
        if position < self.warmup:
            scale = (position + 1) / max(1, self.warmup)
            for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base * scale
            return
        fraction = min(1.0, (position - self.warmup) / max(1, self.total - self.warmup))
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.min_lr + (base - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * fraction))


def _loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shifted_logits = logits[:, :-1].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    count = int((shifted_labels != -100).sum().item())
    if count == 0:
        return logits.float().sum() * 0.0, 0
    return F.cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1), ignore_index=-100, reduction="sum"), count


def _cap_labels(labels: torch.Tensor, *, allowed: int) -> torch.Tensor:
    current = int((labels[:, 1:] != -100).sum().item())
    if current <= allowed:
        return labels
    masked = labels.clone()
    valid = (masked[:, 1:] != -100).nonzero(as_tuple=False)
    if len(valid) > allowed:
        masked[valid[allowed:, 0], valid[allowed:, 1] + 1] = -100
    return masked


@torch.no_grad()
def validate(model, batch_factory: Callable[[int], dict[str, torch.Tensor]], *, batches: int, ctx: dict) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for index in range(batches):
        batch = batch_factory(index)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        value, count = _loss_sum(output.logits, batch["labels"])
        total_loss += float(value.item())
        total_tokens += count
    total_loss = all_reduce_scalar(total_loss, device=ctx["device"])
    total_tokens = int(all_reduce_scalar(total_tokens, device=ctx["device"]))
    if was_training:
        model.train()
    return total_loss / max(1, total_tokens), total_tokens


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    named = list(model.named_parameters())
    return {
        "total": sum(parameter.numel() for _, parameter in named),
        "trainable": sum(parameter.numel() for _, parameter in named if parameter.requires_grad),
        "embedding": sum(parameter.numel() for name, parameter in named if "embed_tokens" in name or "lm_head" in name),
        "transformer": sum(parameter.numel() for name, parameter in named if ".layers." in name),
    }


def save_checkpoint(path: Path, *, model, optimizer, scheduler: CosineSchedule, step: int, tokens_seen: int, config: dict, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(), "scheduler": {"base_lrs": scheduler.base_lrs},
        "step": step, "tokens_seen": tokens_seen, "stage": stage, "config": config,
    }, path)


def load_weights(model, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Checkpoint/model mismatch")
    return payload


def _write_environment(output: Path, ctx: dict) -> None:
    if not rank0(ctx):
        return
    write_json(output / "environment.json", {
        "torch": torch.__version__, "cuda": torch.version.cuda, "world_size": ctx["world_size"],
        "gpu": torch.cuda.get_device_name(ctx["device"]), "gpu_memory_bytes": torch.cuda.get_device_properties(ctx["device"]).total_memory,
    })


def train(
    *,
    method_label: str,
    cfg: dict,
    stage: Literal["pretrain", "math", "multihop"],
    output: Path,
    model_factory: ModelFactory,
    optimizer_factory: OptimizerFactory,
    metric_factory: MetricFactory | None = None,
) -> Path:
    ctx = runtime()
    seed_everything(int(cfg["seed"]), ctx["rank"])
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_factory(tokenizer)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("All trainable parameters must remain FP32 master parameters")
    if rank0(ctx):
        counts = parameter_counts(model)
        if not 90_000_000 <= counts["total"] <= 110_000_000:
            raise RuntimeError(f"Model size is outside 90M-110M: {counts}")
        write_json(output / "parameter_counts.json", counts)
        write_json(output / "config.json", cfg)
    model = wrap_ddp(model, ctx)
    optimizer = optimizer_factory(unwrap(model), cfg, stage)
    train_cfg = cfg["pretrain"] if stage == "pretrain" else cfg["task_training"]
    total = int(train_cfg["steps"] if stage == "pretrain" else train_cfg["token_budget"])
    warmup = int(train_cfg["warmup_steps"] if stage == "pretrain" else total * float(train_cfg["warmup_ratio"]))
    scheduler = CosineSchedule(optimizer, total=total, warmup=warmup, min_lr=float(train_cfg["min_lr"]))
    micro_batch, accumulation = int(train_cfg["micro_batch_per_gpu"]), int(train_cfg["gradient_accumulation"])
    torch.cuda.reset_peak_memory_stats(ctx["device"])
    _write_environment(output, ctx)
    if stage == "pretrain":
        stream = TokenStream(Path(cfg["fineweb"]["train_tokens"]), seq_len=int(train_cfg["seq_len"]))
        val_stream = TokenStream(Path(cfg["fineweb"]["val_tokens"]), seq_len=int(train_cfg["seq_len"]))
        steps = total
        def get_batch(step: int, micro: int):
            return stream.batch(sequence_offset=sequence_offset(step=step, accumulation_index=micro, rank=ctx["rank"], world_size=ctx["world_size"], micro_batch=micro_batch, accumulation=accumulation), batch_size=micro_batch, device=ctx["device"])
        def val_batch(index: int):
            return val_stream.batch(sequence_offset=index * ctx["world_size"] * micro_batch + ctx["rank"] * micro_batch, batch_size=micro_batch, device=ctx["device"])
        validate_batches = int(cfg["fineweb"]["validation_tokens"]) // (int(train_cfg["seq_len"]) * micro_batch * ctx["world_size"])
        budgeted = False
    else:
        examples = encode_examples(tokenizer, load_examples(Path(cfg["task_data"][stage]["train"])), seq_len=int(train_cfg["seq_len"]))
        validation = encode_examples(tokenizer, load_examples(Path(cfg["task_data"][stage]["validation"])), seq_len=int(train_cfg["seq_len"]))
        steps = 10**9
        def get_batch(step: int, micro: int):
            return collate_task(task_examples_for_microbatch(examples, step=step, accumulation_index=micro, rank=ctx["rank"], world_size=ctx["world_size"], micro_batch=micro_batch, accumulation=accumulation), pad_token_id=int(tokenizer.pad_token_id), device=ctx["device"])
        def val_batch(index: int):
            start = (index * ctx["world_size"] + ctx["rank"]) * micro_batch
            return collate_task([validation[(start + offset) % len(validation)] for offset in range(micro_batch)], pad_token_id=int(tokenizer.pad_token_id), device=ctx["device"])
        validate_batches = math.ceil(len(validation) / (ctx["world_size"] * micro_batch))
        budgeted = True
    metrics_path, rolling = output / "metrics.jsonl", deque(maxlen=100)
    tokens_seen, step = 0, 0
    started = time.perf_counter()
    while step < steps and (not budgeted or tokens_seen < total):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum, local_tokens = 0.0, 0
        step_global_tokens = 0
        step_start = time.perf_counter()
        for micro in range(accumulation):
            batch = get_batch(step, micro)
            if budgeted:
                local_count = int((batch["labels"][:, 1:] != -100).sum().item())
                counts = [torch.zeros((), dtype=torch.int64, device=ctx["device"]) for _ in range(ctx["world_size"])]
                dist.all_gather(counts, torch.tensor(local_count, dtype=torch.int64, device=ctx["device"])) if ctx["world_size"] > 1 else counts.__setitem__(0, torch.tensor(local_count, device=ctx["device"]))
                remaining = total - tokens_seen - step_global_tokens
                allowed = max(0, min(local_count, remaining - sum(int(value.item()) for value in counts[:ctx["rank"]])))
                batch["labels"] = _cap_labels(batch["labels"], allowed=allowed)
            sync = micro == accumulation - 1
            context = nullcontext() if sync or not isinstance(model, DDP) else model.no_sync()
            with context, torch.autocast("cuda", dtype=torch.bfloat16):
                output_model = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                loss_sum, count = _loss_sum(output_model.logits, batch["labels"])
                global_count = int(all_reduce_scalar(count, device=ctx["device"]))
                if global_count:
                    (loss_sum * ctx["world_size"] / global_count / accumulation).backward()
            local_loss_sum += float(loss_sum.detach().item())
            local_tokens += count
            step_global_tokens += global_count
        global_loss_sum = all_reduce_scalar(local_loss_sum, device=ctx["device"])
        global_tokens = int(all_reduce_scalar(local_tokens, device=ctx["device"]))
        if global_tokens == 0:
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        optimizer.step()
        tokens_seen += global_tokens if budgeted else int(ctx["world_size"] * micro_batch * accumulation * train_cfg["seq_len"])
        scheduler.step(tokens_seen if budgeted else step + 1)
        raw_loss = global_loss_sum / global_tokens
        rolling.append(raw_loss)
        query_norm, query_grad_norm = metric_factory(unwrap(model)) if metric_factory else (None, None)
        should_validate = (step + 1) % int(train_cfg["validation_interval_steps"]) == 0 or (budgeted and tokens_seen >= total) or (not budgeted and step + 1 == steps)
        validation_loss, validation_ppl = None, None
        if should_validate:
            validation_loss, _ = validate(model, val_batch, batches=validate_batches, ctx=ctx)
            validation_ppl = float(math.exp(validation_loss))
        peak_allocated = all_reduce_scalar(torch.cuda.max_memory_allocated(ctx["device"]) / 2**30, device=ctx["device"], op=dist.ReduceOp.MAX)
        peak_reserved = all_reduce_scalar(torch.cuda.max_memory_reserved(ctx["device"]) / 2**30, device=ctx["device"], op=dist.ReduceOp.MAX)
        elapsed = time.perf_counter() - step_start
        record = {
            "method": method_label, "stage": stage, "task": None if stage == "pretrain" else stage, "step": step + 1,
            "tokens_seen": tokens_seen, "train_loss_raw": raw_loss, "train_loss_100step_mean": sum(rolling) / len(rolling),
            "validation_loss": validation_loss, "validation_ppl": validation_ppl,
            "task_metric": None,
            "learning_rates": [group["lr"] for group in optimizer.param_groups], "query_norm": query_norm,
            "query_grad_norm": query_grad_norm, "peak_allocated_gb": peak_allocated, "peak_reserved_gb": peak_reserved,
            "tokens_per_second": global_tokens / elapsed, "step_time": elapsed, "wall_clock": time.perf_counter() - started,
        }
        if rank0(ctx):
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        step += 1
        if (stage == "pretrain" and step % 500 == 0) or (budgeted and step % int(train_cfg["checkpoint_interval_steps"]) == 0):
            if rank0(ctx):
                save_checkpoint(output / "latest.pt", model=model, optimizer=optimizer, scheduler=scheduler, step=step, tokens_seen=tokens_seen, config=cfg, stage=stage)
        barrier(ctx)
    if rank0(ctx):
        save_checkpoint(output / "final.pt", model=model, optimizer=optimizer, scheduler=scheduler, step=step, tokens_seen=tokens_seen, config=cfg, stage=stage)
        rows = read_jsonl(metrics_path)
        write_json(output / "run_manifest.json", {"method": method_label, "stage": stage, "steps": step, "tokens_seen": tokens_seen, "micro_batch_per_gpu": micro_batch, "gradient_accumulation": accumulation, "world_size": ctx["world_size"], "effective_global_batch_sequences": ctx["world_size"] * micro_batch * accumulation, "tokens_per_optimizer_step": ctx["world_size"] * micro_batch * accumulation * int(train_cfg["seq_len"]), "metrics_sha256": sha256_file(metrics_path), "final_checkpoint": str(output / "final.pt")})
        summary = {"final_train_loss": rows[-1]["train_loss_raw"], "final_validation_loss": rows[-1]["validation_loss"], "peak_allocated_gb": max(row["peak_allocated_gb"] for row in rows), "average_tokens_per_second": sum(row["tokens_per_second"] for row in rows) / len(rows)}
        if stage in {"math", "multihop"}:
            summary["task_metric"] = task_metrics(unwrap(model), tokenizer, load_examples(Path(cfg["task_data"][stage]["validation"])), task=stage, device=ctx["device"])
        write_json(output / "metrics_summary.json", summary)
    barrier(ctx)
    return output / "final.pt"
