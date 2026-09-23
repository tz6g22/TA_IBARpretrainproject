from __future__ import annotations

from .modeling import Qwen3KimiAttnResForCausalLM, config_from_pretrained


def full_to_task_block(full_model: Qwen3KimiAttnResForCausalLM, partition: list[int]) -> Qwen3KimiAttnResForCausalLM:
    """Retain the Main Full checkpoint parameters while changing source grouping."""
    if full_model.config.kimi_mode != "full" or sum(partition) != full_model.config.num_hidden_layers:
        raise ValueError("Invalid TA-IBAR Full-to-Block conversion")
    target = Qwen3KimiAttnResForCausalLM(
        config_from_pretrained(full_model.config, mode="block", block_sizes=partition)
    )
    incompatible = target.load_state_dict(full_model.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("TA-IBAR conversion changed checkpoint parameters")
    target.config.use_cache = False
    return target
