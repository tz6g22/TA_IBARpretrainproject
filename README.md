# TA-IBAR Train from Scratch

This project is independent of `MoiraiBlockPost_train` at runtime. It trains a
12-layer Qwen3-style model from random initialization on a fixed FineWeb
sample-10BT token stream, then compares Native, Kimi Full, Kimi Block and
TA-IBAR.

Run the one-time task-data migration before the GPU workflow:

```bash
python scripts/import_posttrain_task_data.py
python scripts/prepare_fineweb.py
PYTHON_BIN=/path/to/gpu/venv/bin/python bash scripts/start_formal.sh
```

`start_formal.sh` executes the sole integrated smoke test, publishes the
reproducible source if GitHub CLI authentication is available, then detaches
the formal pipeline with `nohup`. The pipeline hard-stops after both Main
Discovery tau sweeps and never starts Main task training without a selected
tau.

The shared-memory L4 configuration uses micro_batch_per_gpu=2 and
gradient_accumulation=8, preserving the 64-sequence effective global batch and
131072 FineWeb tokens per optimizer step.
