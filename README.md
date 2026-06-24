# DLMon

Runtime correctness monitor for PyTorch data pipelines. Attaches to
DataLoaders and checks for silent bugs: problems that don't crash but
corrupt training results (broken shuffle, duplicate samples, frozen
augmentation, NaN injection, dtype truncation, etc.).

## Install

```bash
pip install git+https://github.com/ArkaD171717/DLMon.git
```

Requires Python 3.10+, PyTorch 2.2+, xxhash. Torchvision optional (needed
for some transform-level checks).

**Tested PyTorch versions:** 2.2, 2.4, 2.5, latest (CI matrix covers
Python 3.10–3.13 on Ubuntu and Windows).

## Usage

```python
from dlmon import DLMonitor

monitor = DLMonitor()
train_guarded = monitor.guard(train_loader, role="train")
val_guarded = monitor.guard(val_loader, role="val")

for epoch in range(num_epochs):
    for batch in train_guarded:
        ...
    for batch in val_guarded:
        ...

print(monitor.summary())
violations = monitor.get_violations()
```

`guard()` returns a `GuardedLoader` wrapper. Iterate the returned object
instead of the raw DataLoader. Each violation is a `Violation` object
with `.type` (a `ViolationType` enum), `.message`, `.epoch`, `.batch_idx`,
and `.details`.

### Pre-training transform checks

`preflight()` runs transform checks in the main process before training
starts. Works regardless of `num_workers`:

```python
result = monitor.preflight(dataset, n=50, repeats=3)
if result["violations"]:
    print("Transform issues found:", result["violations"])
```

### Partition overlap

```python
train_guarded = monitor.guard(train_loader, role="train")
val_guarded = monitor.guard(val_loader, role="val")
overlaps = monitor.check_partitions()
```

## What it checks

### Sampler-level (any `num_workers`, any dataset)

| Check | ViolationType | Confidence | What it catches |
|-------|--------------|------------|----------------|
| Sample uniqueness | `DUPLICATE_SAMPLE` | Exact | Same sample index seen twice in one epoch |
| Epoch completeness | `MISSING_SAMPLE` | Exact | Samples missing from an epoch |
| Shuffle detection | `NO_SHUFFLE` | Statistical | Identical sample ordering across consecutive epochs |
| Sequential order | `UNEXPECTED_SHUFFLE` | Statistical | Ordering changed when it shouldn't (val/test loaders) |
| Partition overlap | `PARTITION_OVERLAP` | Exact | Train/val/test index sets share samples |
| Batch distribution | `BATCH_DISTRIBUTION_OOB` | Statistical | Per-batch class counts outside learned bounds |

### Batch-level (any `num_workers`)

These checks run inside `GuardedLoader` on collated batch tensors. They
work with any `num_workers` because they inspect batch output, not
per-sample `__getitem__`.

| Check | ViolationType | Confidence | What it catches |
|-------|--------------|------------|----------------|
| NaN/Inf detection | `NAN_INF_OUTPUT` | Exact | NaN or Inf values in batch tensors |
| Shape consistency | `SHAPE_MISMATCH` | Exact | Non-batch dimensions or dtype changed across batches |
| Collate corruption | `COLLATE_LABEL_CORRUPTION` | Exact | Custom `collate_fn` that reorders or drops labels |

### Dataset-level (`num_workers=0` only)

These checks intercept `__getitem__` in the main process. The first 64
calls per loader are always monitored, then every 100th call thereafter.
Pervasive bugs are reliably caught; single-index defects may be missed.
`preflight()` checks every sampled index.

| Check | ViolationType | Confidence | What it catches |
|-------|--------------|------------|----------------|
| Transform no-op | `TRANSFORM_NOOP` | Heuristic | Transform produces identical output to input |
| Transform absent | `TRANSFORM_NEVER_INVOKED` | Exact | Dataset has `.transform` but never calls it |
| Dtype truncation | `DTYPE_TRUNCATION` | Exact | Precision loss in transform chain (e.g. float32→float16) |
| Label cross-check | `LABEL_MISMATCH` | Exact | Returned label differs from cached `.targets` |
| Frozen stochastic | `FROZEN_STOCHASTIC` | Heuristic | Stochastic transform producing identical outputs |

### Guard-time (checked once at `guard()` call, any `num_workers`)

| Check | ViolationType | Confidence | What it catches |
|-------|--------------|------------|----------------|
| Stochastic eval | `STOCHASTIC_EVAL_TRANSFORM` | Heuristic | Random transforms on val/test loaders |
| Worker seed collision | `FROZEN_WORKER_SEEDS` | Heuristic | `worker_init_fn` that freezes all workers to the same seed |

### Confidence levels

- **Exact**: deterministic detection with zero false positives. The check
  sees the full data (all indices, all tensors) and reports only provable
  violations.
- **Statistical**: compares observations across epochs or batches. May
  miss violations that affect only a small window, or may flag rare
  coincidences. False positive rate is low but nonzero.
- **Heuristic**: uses proxy signals (output equality, attribute names,
  hash collisions). Can produce false positives on unusual pipelines.
  Treat as a warning to investigate, not proof of a bug.

## `num_workers > 0`

With `num_workers > 0`:

- **Sampler-level checks** remain fully active (they intercept sampler
  iteration in the main process)
- **Batch-level checks** (NaN, shape, collate) remain active via
  `GuardedLoader`
- **Guard-time checks** (stochastic eval, worker seeds) remain active
- **Dataset-level checks** are disabled (a warning is emitted); use
  `preflight()` for pre-training transform validation

Collate label checking is disabled with `num_workers > 0` because worker
prefetching desynchronizes batch delivery order from sampler-tracked
indices.

## Notes

Claude used for formatting, tweaks, and testing.
