# Experiment Plan

FineWeb pretraining uses 4 GPUs, sequence length 2048, micro-batch 2 and
gradient accumulation 8. This is 64 sequences and 131072 input tokens per
optimizer step; 2500 steps therefore consume 327680000 input tokens. This
micro-batch adjustment accommodates shared L4 memory; the effective global
batch and token budget are unchanged.

The formal pipeline is Native pretrain and two task branches, Kimi Full
pretrain and two task branches, Kimi Block pretrain and two task branches,
then Main Full pretrain and task-specific Linear CKA-min DP discovery. Main
task training is intentionally disabled until a tau is selected.
