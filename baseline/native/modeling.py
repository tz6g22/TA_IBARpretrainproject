from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


TINY_QWEN3 = {
    "num_hidden_layers": 12,
    "hidden_size": 448,
    "intermediate_size": 1344,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "attention_dropout": 0.0,
    "tie_word_embeddings": True,
    "rms_norm_eps": 1.0e-6,
    "rope_theta": 1_000_000.0,
    "use_cache": False,
}


def build_model(tokenizer, architecture: dict) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        pad_token_id=int(tokenizer.pad_token_id),
        bos_token_id=int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        **TINY_QWEN3,
    )
    config._attn_implementation = "sdpa"
    for key in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "max_position_embeddings"):
        if int(getattr(config, key)) != int(architecture[key]):
            raise RuntimeError(f"Native architecture mismatch for {key}")
    return Qwen3ForCausalLM(config)


def optimizer_factory(model: torch.nn.Module, cfg: dict, stage: str) -> torch.optim.Optimizer:
    options = cfg["optimizer"] if stage == "pretrain" else cfg["task_optimizer"]
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any("pseudo_query" in name or "key_norm" in name for name in names):
        raise RuntimeError("Native model must not contain AttnRes parameters")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(options["backbone_lr"]),
        weight_decay=float(options["backbone_weight_decay"]),
        betas=tuple(options["betas"]),
        eps=float(options["eps"]),
    )
