# Project State

Implementation is self-contained. Runtime files are generated only under this
project. `scripts/import_posttrain_task_data.py` is a one-time migration tool;
the formal pipeline does not import code or data from the Post-training repo.

Current execution prerequisite: an SSH node where four CUDA L4 GPUs are visible
to PyTorch, a prepared FineWeb sample-10BT stream, and authenticated GitHub CLI.

All methods use micro_batch_per_gpu=2 and gradient_accumulation=8. With four
GPUs and seq_len=2048 this preserves 64 sequences and 131072 tokens per
optimizer step, for 327680000 FineWeb tokens across 2500 steps.
