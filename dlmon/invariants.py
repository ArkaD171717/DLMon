from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import xxhash

CHECK_WINDOW = 64


class ViolationType(Enum):
    DUPLICATE_SAMPLE = auto()
    MISSING_SAMPLE = auto()
    PARTITION_OVERLAP = auto()
    BATCH_DISTRIBUTION_OOB = auto()
    TRANSFORM_NOOP = auto()
    FROZEN_STOCHASTIC = auto()
    LABEL_MISMATCH = auto()
    NO_SHUFFLE = auto()
    TRANSFORM_NEVER_INVOKED = auto()
    UNEXPECTED_SHUFFLE = auto()
    SHAPE_MISMATCH = auto()
    NAN_INF_OUTPUT = auto()
    STOCHASTIC_EVAL_TRANSFORM = auto()
    DTYPE_TRUNCATION = auto()
    FROZEN_WORKER_SEEDS = auto()
    COLLATE_LABEL_CORRUPTION = auto()


@dataclass
class Violation:
    type: ViolationType
    message: str
    epoch: int = -1
    batch_idx: int = -1
    sample_ids: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.type.name}] epoch={self.epoch} batch={self.batch_idx}: {self.message}"


class SampleUniqueness:

    def __init__(self, dup_tolerance: int = 0):
        self.dup_tolerance = dup_tolerance
        self.seen: set[int] = set()
        self.dup_count: int = 0
        self.violations: list[Violation] = []
        self.epoch = 0

    def start_epoch(self, epoch: int):
        self.seen.clear()
        self.dup_count = 0
        self.epoch = epoch

    def check_batch(self, sample_ids: list[int], batch_idx: int) -> list[Violation]:
        new_violations = []
        for sid in sample_ids:
            h = xxhash.xxh64_intdigest(sid.to_bytes(8, "little"))
            if h in self.seen:
                self.dup_count += 1
                if self.dup_count > self.dup_tolerance:
                    v = Violation(
                        type=ViolationType.DUPLICATE_SAMPLE,
                        message=f"Sample {sid} already seen this epoch",
                        epoch=self.epoch,
                        batch_idx=batch_idx,
                        sample_ids=[sid],
                    )
                    new_violations.append(v)
                    self.violations.append(v)
            self.seen.add(h)
        return new_violations


class CompletenessMode(Enum):
    EXACT_SET = auto()
    COUNT_ONLY = auto()
    OFF = auto()


class EpochCompleteness:

    def __init__(
        self,
        dataset_size: int,
        mode: CompletenessMode = CompletenessMode.EXACT_SET,
        expected_set: set[int] | None = None,
        expected_count: int | None = None,
    ):
        self.dataset_size = dataset_size
        self.mode = mode
        self.expected_set = expected_set
        self.expected_count = expected_count
        self.seen_ids: set[int] = set()
        self.violations: list[Violation] = []
        self.epoch = 0

    def start_epoch(self, epoch: int):
        self.seen_ids.clear()
        self.epoch = epoch

    def record_sample(self, sample_id: int):
        self.seen_ids.add(sample_id)

    def end_epoch(self) -> list[Violation]:
        if self.mode is CompletenessMode.OFF:
            return []

        new_violations = []

        if self.mode is CompletenessMode.EXACT_SET:
            expected = self.expected_set if self.expected_set is not None else set(range(self.dataset_size))
            missing = expected - self.seen_ids
            if missing:
                v = Violation(
                    type=ViolationType.MISSING_SAMPLE,
                    message=f"{len(missing)} samples not seen in epoch "
                            f"(expected {len(expected)}, got {len(self.seen_ids)})",
                    epoch=self.epoch,
                    sample_ids=sorted(list(missing))[:20],
                    details={"missing_count": len(missing),
                             "seen_count": len(self.seen_ids)},
                )
                new_violations.append(v)
                self.violations.append(v)

        elif self.mode is CompletenessMode.COUNT_ONLY:
            count = self.expected_count if self.expected_count is not None else self.dataset_size
            if len(self.seen_ids) != count:
                v = Violation(
                    type=ViolationType.MISSING_SAMPLE,
                    message=f"Expected {count} unique samples, got {len(self.seen_ids)}",
                    epoch=self.epoch,
                    details={"expected_count": count,
                             "seen_count": len(self.seen_ids)},
                )
                new_violations.append(v)
                self.violations.append(v)

        return new_violations


class PartitionExclusive:

    def __init__(self):
        self.partitions: dict[str, set[int]] = {}
        self.violations: list[Violation] = []

    def register(self, name: str, sample_ids):
        self.partitions[name] = set(sample_ids)

    def check(self) -> list[Violation]:
        new_violations = []
        names = list(self.partitions.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = self.partitions[names[i]] & self.partitions[names[j]]
                if overlap:
                    v = Violation(
                        type=ViolationType.PARTITION_OVERLAP,
                        message=f"{len(overlap)} samples in both "
                                f"'{names[i]}' and '{names[j]}'",
                        sample_ids=sorted(list(overlap))[:20],
                        details={
                            "partition_a": names[i],
                            "partition_b": names[j],
                            "overlap_count": len(overlap),
                        },
                    )
                    new_violations.append(v)
                    self.violations.append(v)
        return new_violations


class BatchDistribution:
    # sigma floored at sqrt(mean) to prevent zero-width bounds from
    # degenerate learning epochs (sequential sampling, sigma=0)

    def __init__(self, sigma_factor: float = 5.0):
        self.sigma_factor = sigma_factor
        self.learning = True
        self.learned_bounds: dict[str, tuple[float, float]] = {}
        self._learning_stats: dict[str, list[float]] = {}
        self.violations: list[Violation] = []
        self.epoch = 0

    def start_epoch(self, epoch: int):
        self.epoch = epoch
        if epoch > 0 and self.learning:
            self._finalize_learning()
            self.learning = False

    def _finalize_learning(self):
        for key, vals in self._learning_stats.items():
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
            sigma = max(var ** 0.5, max(abs(mean), 1.0) ** 0.5)
            margin = self.sigma_factor * sigma
            self.learned_bounds[key] = (mean - margin, mean + margin)

    def observe(self, key: str, value: float, batch_idx: int) -> list[Violation]:
        if self.learning:
            self._learning_stats.setdefault(key, []).append(value)
            return []

        if key in self.learned_bounds:
            lo, hi = self.learned_bounds[key]
            if value < lo or value > hi:
                v = Violation(
                    type=ViolationType.BATCH_DISTRIBUTION_OOB,
                    message=f"'{key}' = {value:.4f} outside learned "
                            f"bounds [{lo:.4f}, {hi:.4f}]",
                    epoch=self.epoch,
                    batch_idx=batch_idx,
                    details={"key": key, "value": value,
                             "lo": lo, "hi": hi},
                )
                self.violations.append(v)
                return [v]
        return []


class SampleFlow:

    def __init__(self):
        self.constraints: list[dict] = []
        self.violations: list[Violation] = []
        self.epoch = 0

    def add_shape_constraint(self, transform_name: str,
                              input_shape_fn, output_shape_fn):
        self.constraints.append({
            "name": transform_name,
            "input_fn": input_shape_fn,
            "output_fn": output_shape_fn,
        })

    def check(self, transform_name: str, input_tensor, output_tensor,
              batch_idx: int = -1) -> list[Violation]:
        new_violations = []
        for c in self.constraints:
            if c["name"] == transform_name:
                expected = c["input_fn"](input_tensor)
                actual = c["output_fn"](output_tensor)
                if expected != actual:
                    v = Violation(
                        type=ViolationType.SHAPE_MISMATCH,
                        message=f"Transform '{transform_name}': expected "
                                f"output shape {expected}, got {actual}",
                        epoch=self.epoch,
                        batch_idx=batch_idx,
                        details={"transform": transform_name,
                                 "expected": str(expected),
                                 "actual": str(actual)},
                    )
                    new_violations.append(v)
                    self.violations.append(v)
        return new_violations


class ShuffleDetector:

    def __init__(self, check_size: int = CHECK_WINDOW):
        self.check_size = check_size
        self.prev_order: list[int] | None = None
        self.current_order: list[int] = []
        self.violations: list[Violation] = []
        self.epoch = 0

    def start_epoch(self, epoch: int):
        self.epoch = epoch
        if self.current_order:
            self.prev_order = self.current_order[:]
        self.current_order = []

    def record_samples(self, sample_ids: list[int]):
        remaining = self.check_size - len(self.current_order)
        if remaining > 0:
            self.current_order.extend(sample_ids[:remaining])

    def end_epoch(self) -> list[Violation]:
        if self.prev_order is not None and len(self.current_order) >= self.check_size:
            prev = self.prev_order[:self.check_size]
            curr = self.current_order[:self.check_size]
            if prev == curr:
                v = Violation(
                    type=ViolationType.NO_SHUFFLE,
                    message=f"First {self.check_size} samples identical "
                            f"across epochs {self.epoch-1} and {self.epoch}; "
                            f"shuffling may be broken",
                    epoch=self.epoch,
                    details={"check_size": self.check_size},
                )
                self.violations.append(v)
                return [v]
        return []


class SequentialOrderDetector:

    def __init__(self, check_size: int = CHECK_WINDOW):
        self.check_size = check_size
        self.prev_order: list[int] | None = None
        self.current_order: list[int] = []
        self.violations: list[Violation] = []
        self.epoch = 0

    def start_epoch(self, epoch: int):
        self.epoch = epoch
        if self.current_order:
            self.prev_order = self.current_order[:]
        self.current_order = []

    def record_samples(self, sample_ids: list[int]):
        remaining = self.check_size - len(self.current_order)
        if remaining > 0:
            self.current_order.extend(sample_ids[:remaining])

    def end_epoch(self) -> list[Violation]:
        new_violations = []
        if self.prev_order is not None and len(self.current_order) >= self.check_size:
            prev = self.prev_order[:self.check_size]
            curr = self.current_order[:self.check_size]
            if prev != curr:
                v = Violation(
                    type=ViolationType.UNEXPECTED_SHUFFLE,
                    message=f"First {self.check_size} samples differ across "
                            f"epochs {self.epoch-1} and {self.epoch}; "
                            f"ordering changed unexpectedly for a sequential loader",
                    epoch=self.epoch,
                    details={"check_size": self.check_size},
                )
                new_violations.append(v)
                self.violations.append(v)
        return new_violations


class TransformApplied:

    def __init__(self, noop_threshold: float = 0.95):
        self.noop_threshold = noop_threshold
        self.total_checks = 0
        self.noop_count = 0
        self.violations: list[Violation] = []
        self.epoch = 0

    def start_epoch(self, epoch: int):
        self.epoch = epoch
        self.total_checks = 0
        self.noop_count = 0

    def check(self, transform_name: str, input_data, output_data,
              batch_idx: int = -1) -> bool:
        import torch

        self.total_checks += 1
        if isinstance(input_data, torch.Tensor) and isinstance(output_data, torch.Tensor):
            if torch.equal(input_data, output_data):
                self.noop_count += 1
        return self.noop_count > 0

    def end_epoch(self) -> list[Violation]:
        if self.total_checks == 0:
            return []
        noop_rate = self.noop_count / self.total_checks
        if noop_rate >= self.noop_threshold:
            v = Violation(
                type=ViolationType.TRANSFORM_NOOP,
                message=f"Transform is a no-op on {noop_rate*100:.0f}% "
                        f"of {self.total_checks} samples",
                epoch=self.epoch,
                details={"noop_rate": noop_rate,
                         "noop_count": self.noop_count,
                         "total": self.total_checks},
            )
            self.violations.append(v)
            return [v]
        return []


class StochasticDiversity:

    def __init__(self, min_unique_ratio: float = 0.1):
        self.min_unique_ratio = min_unique_ratio
        self.output_hashes: list[int] = []
        self._prev_epoch_hashes: set[int] | None = None
        self.violations: list[Violation] = []
        self.epoch = 0

    def start_epoch(self, epoch: int):
        self.epoch = epoch
        if self.output_hashes:
            self._prev_epoch_hashes = set(self.output_hashes)
        self.output_hashes.clear()

    def record_output(self, output_data) -> None:
        import torch

        if isinstance(output_data, torch.Tensor):
            data_bytes = output_data.detach().cpu().numpy().tobytes()
            self.output_hashes.append(xxhash.xxh64_intdigest(data_bytes))

    def end_epoch(self) -> list[Violation]:
        new_violations = []
        if len(self.output_hashes) >= 10:
            unique = len(set(self.output_hashes))
            ratio = unique / len(self.output_hashes)
            if ratio < self.min_unique_ratio:
                v = Violation(
                    type=ViolationType.FROZEN_STOCHASTIC,
                    message=f"Only {unique}/{len(self.output_hashes)} unique "
                            f"outputs ({ratio*100:.1f}%); random seed may be frozen",
                    epoch=self.epoch,
                    details={"unique": unique,
                             "total": len(self.output_hashes),
                             "ratio": ratio,
                             "scope": "within_epoch"},
                )
                new_violations.append(v)
                self.violations.append(v)

            if self._prev_epoch_hashes is not None:
                current_set = set(self.output_hashes)
                overlap = len(current_set & self._prev_epoch_hashes)
                overlap_ratio = overlap / max(len(current_set), 1)
                if overlap_ratio > 0.9:
                    v = Violation(
                        type=ViolationType.FROZEN_STOCHASTIC,
                        message=f"{overlap_ratio*100:.0f}% of outputs identical "
                                f"to previous epoch; RNG may not be reseeding "
                                f"between epochs",
                        epoch=self.epoch,
                        details={"overlap": overlap,
                                 "current_unique": len(current_set),
                                 "prev_unique": len(self._prev_epoch_hashes),
                                 "overlap_ratio": overlap_ratio,
                                 "scope": "cross_epoch"},
                    )
                    new_violations.append(v)
                    self.violations.append(v)

        return new_violations
