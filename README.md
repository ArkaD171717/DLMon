# DLMon

Runtime correctness monitor for PyTorch data pipelines. Attaches to
DataLoaders and checks for silent bugs: problems that don't crash but
corrupt training results (broken shuffle, duplicate samples, frozen
augmentation, NaN injection, dtype truncation, etc.).

## Install

```bash
pip install git+https://github.com/ArkaD171717/DLMon.git
```

Requires Python 3.10+, PyTorch 2.0+, xxhash. Torchvision optional (needed
for some transform-level checks).

## Usage

```python
from dlmon import DLMonitor
from torch.utils.data import DataLoader

monitor = DLMonitor()
monitor.guard(train_loader, role="train")
monitor.guard(val_loader, role="val")

for epoch in range(num_epochs):
    for batch in train_loader:
        ...
    for batch in val_loader:
        ...

print(monitor.summary())
violations = monitor.get_violations()
```

`guard()` wraps the loader's sampler in place and returns the same loader
object. Call it before iterating. Each violation is a `Violation` object
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
monitor.guard(train_loader, role="train")
monitor.guard(val_loader, role="val")
overlaps = monitor.check_partitions()
```

## What it checks

### Sampler-level (any `num_workers`, any dataset)

| Check | ViolationType | What it catches |
|-------|--------------|----------------|
| Sample uniqueness | `DUPLICATE_SAMPLE` | Same sample index seen twice in one epoch |
| Epoch completeness | `MISSING_SAMPLE` | Samples missing from an epoch |
| Shuffle detection | `NO_SHUFFLE` | Identical sample ordering across consecutive epochs |
| Sequential order | `UNEXPECTED_SHUFFLE` | Ordering changed when it shouldn't (val/test loaders) |
| Partition overlap | `PARTITION_OVERLAP` | Train/val/test index sets share samples |
| Batch distribution | `BATCH_DISTRIBUTION_OOB` | Per-batch class counts outside learned bounds |

### Dataset-level (`num_workers=0` only)

These checks use sampling: the first 64 `__getitem__` calls per loader are
always monitored, then every 100th call thereafter. Pervasive bugs (affecting
all samples) are reliably caught; single-index defects may be missed
depending on alignment. `preflight()` checks every sampled index.

| Check | ViolationType | What it catches |
|-------|--------------|----------------|
| Transform no-op | `TRANSFORM_NOOP` | Transform produces identical output to input |
| Transform absent | `TRANSFORM_NEVER_INVOKED` | Dataset has `.transform` but never calls it |
| NaN/Inf detection | `NAN_INF_OUTPUT` | NaN or Inf values in dataset output |
| Dtype truncation | `DTYPE_TRUNCATION` | Precision loss in transform chain (e.g. float32 to float16) |
| Label cross-check | `LABEL_MISMATCH` | Returned label differs from cached `.targets` |
| Frozen stochastic | `FROZEN_STOCHASTIC` | Stochastic transform producing identical outputs |
| Shape consistency | `SHAPE_MISMATCH` | Shape/dtype varies across repeated calls to same index |

### Guard-time (checked once at `guard()` call, any `num_workers`)

| Check | ViolationType | What it catches |
|-------|--------------|----------------|
| Stochastic eval | `STOCHASTIC_EVAL_TRANSFORM` | Random transforms on val/test loaders |
| Worker seed collision | `FROZEN_WORKER_SEEDS` | `worker_init_fn` that freezes all workers to the same seed |
| Collate corruption | `COLLATE_LABEL_CORRUPTION` | Custom `collate_fn` that reorders or drops labels |

## `num_workers > 0` limitation

Dataset-level checks (NaN, dtype, transform no-op, label cross-check)
require `num_workers=0` because `DatasetGuard` cannot intercept
`__getitem__` calls in worker subprocesses. With `num_workers > 0`:

- Sampler-level checks remain fully active
- Guard-time checks (stochastic eval, worker seeds, collate) remain active
- Dataset-level checks are disabled (a warning is emitted)
- Use `preflight()` for pre-training transform validation

## Notes

Claude used for formatting, tweaks, and testing.
