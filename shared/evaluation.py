from __future__ import annotations

import collections
import re
import string

import torch


def _normalize(value: str) -> str:
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(value.split())


def _math_answer(value: str) -> str:
    matches = re.findall(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)", value.rsplit("####", 1)[-1])
    return matches[-1].replace(",", "") if matches else ""


@torch.no_grad()
def task_metrics(model, tokenizer, examples, *, task: str, device: torch.device, max_new_tokens: int = 32) -> dict[str, float]:
    """Greedy, cache-free task evaluation matching the no-cache training model."""
    model.eval()
    scores = []
    for example in examples:
        ids = torch.tensor([tokenizer.encode(example.prompt, add_special_tokens=False)], device=device)
        for _ in range(max_new_tokens):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            ids = torch.cat((ids, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        prediction = tokenizer.decode(ids[0, -max_new_tokens:], skip_special_tokens=True)
        if task == "math":
            scores.append(float(bool(_math_answer(prediction)) and _math_answer(prediction) == _math_answer(example.target)))
        else:
            predicted, gold = _normalize(prediction).split(), _normalize(example.target).split()
            common = sum((collections.Counter(predicted) & collections.Counter(gold)).values())
            precision = common / len(predicted) if predicted else 0.0
            recall = common / len(gold) if gold else 0.0
            scores.append((float(predicted == gold), 2 * precision * recall / (precision + recall) if precision + recall else 0.0))
    if task == "math":
        return {"accuracy": sum(scores) / len(scores)}
    return {"em": sum(item[0] for item in scores) / len(scores), "f1": sum(item[1] for item in scores) / len(scores)}
