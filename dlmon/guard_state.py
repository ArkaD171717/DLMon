from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dlmon.invariants import (
    CHECK_WINDOW,
    BatchDistribution,
    CompletenessMode,
    EpochCompleteness,
    SampleUniqueness,
    SequentialOrderDetector,
    ShuffleDetector,
    Violation,
    ViolationType,
)

if TYPE_CHECKING:
    from dlmon.sampler_guard import SamplerSpec

logger = logging.getLogger("dlmon")


class GuardState:

    def __init__(self, spec: SamplerSpec, dataset_len: int, role: str = "train"):
        self.spec = spec
        self.dataset_len = dataset_len
        self.role = role
        self.epoch: int = 0
        self.violations: list[Violation] = []
        self.seen_indices: set[int] = set()
        self.all_seen_indices: set[int] = set()
        self.dataset_ref = None
        self.dataset_monitoring_status: str = "NOT_CONFIGURED"
        self._getitem_count: int = 0

        self.label_cache: list[int] | None = None
        self.batch_dist: BatchDistribution | None = None
        self.label_cache_status: str = "NOT_CONFIGURED"

        self.monitored_compose = None
        self._transform_noop_fired: set[int] = set()
        self._transform_never_invoked_fired: set[int] = set()
        self._label_mismatch_fired: set[int] = set()
        self.label_check_enabled: bool = False

        self._batch_size: int | None = None
        self._batch_accumulator: list[int] = []
        self._batch_count: int = 0

        self.iterable_dup_threshold: float = 0.05
        self.iterable_min_samples: int = 1000

        self.uniqueness: SampleUniqueness | None = None
        if spec.uniqueness:
            self.uniqueness = SampleUniqueness(dup_tolerance=spec.dup_tolerance)

        self.completeness: EpochCompleteness | None = None
        if spec.completeness is not CompletenessMode.OFF:
            self.completeness = EpochCompleteness(
                dataset_size=dataset_len,
                mode=spec.completeness,
                expected_set=spec.expected_set,
                expected_count=spec.expected_count,
            )

        self.shuffle: ShuffleDetector | None = None
        if spec.shuffle_expected and role == "train":
            self.shuffle = ShuffleDetector()

        self.sequential_order: SequentialOrderDetector | None = None
        if spec.sequential_provenance == "structural" and role != "train":
            self.sequential_order = SequentialOrderDetector()

    def on_epoch_boundary(self) -> None:
        self.epoch += 1
        self.seen_indices.clear()
        self._batch_accumulator.clear()
        self._batch_count = 0

        if self.uniqueness is not None:
            self.uniqueness.start_epoch(self.epoch)
        if self.completeness is not None:
            self.completeness.start_epoch(self.epoch)
        if self.shuffle is not None:
            self.shuffle.start_epoch(self.epoch)
        if self.sequential_order is not None:
            self.sequential_order.start_epoch(self.epoch)
        if self.batch_dist is not None:
            # BatchDistribution uses epoch 0 for learning. GuardState.epoch is
            # already incremented (1-based), so pass epoch - 1 for 0-based.
            self.batch_dist.start_epoch(self.epoch - 1)

    def on_index(self, idx: int) -> None:
        self.seen_indices.add(idx)
        self.all_seen_indices.add(idx)

        if self.uniqueness is not None:
            vs = self.uniqueness.check_batch([idx], batch_idx=-1)
            self.violations.extend(vs)

        if self.completeness is not None:
            self.completeness.record_sample(idx)

        if self.shuffle is not None:
            self.shuffle.record_samples([idx])

        if self.sequential_order is not None:
            self.sequential_order.record_samples([idx])

        # Auto-batching batch boundary reconstruction for BatchDistribution
        if self._batch_size is not None and self.label_cache is not None:
            self._batch_accumulator.append(idx)
            if len(self._batch_accumulator) >= self._batch_size:
                self._flush_batch_accumulator()

    def _flush_batch_accumulator(self) -> None:
        if self._batch_accumulator and self.label_cache is not None and self.batch_dist is not None:
            self._batch_count += 1
            if self._batch_size is None or len(self._batch_accumulator) == self._batch_size:
                self._feed_batch_dist(self._batch_accumulator)
            self._batch_accumulator = []

    def on_batch_boundary(self, indices: list[int]) -> None:
        if self.label_cache is not None and self.batch_dist is not None:
            self._batch_count += 1
            self._feed_batch_dist(indices)

    def _feed_batch_dist(self, indices: list[int]) -> None:
        class_counts: dict[int, int] = {}
        for idx in indices:
            if 0 <= idx < len(self.label_cache):
                cls = self.label_cache[idx]
                class_counts[cls] = class_counts.get(cls, 0) + 1
        for cls, count in class_counts.items():
            vs = self.batch_dist.observe(
                f"class_{cls}_count", float(count), self._batch_count
            )
            self.violations.extend(vs)

    def on_iterable_epoch_end(self, n: int, dups: int) -> None:
        if n >= self.iterable_min_samples and n > 0:
            ratio = dups / n
            if ratio >= self.iterable_dup_threshold:
                v = Violation(
                    type=ViolationType.DUPLICATE_SAMPLE,
                    message=f"IterableDataset: {dups}/{n} duplicates "
                            f"({ratio*100:.1f}%) exceeds {self.iterable_dup_threshold*100:.0f}% threshold",
                    epoch=self.epoch,
                    details={"n": n, "dups": dups, "ratio": ratio},
                )
                self.violations.append(v)

    def on_epoch_close(self, complete: bool) -> None:
        # Flush any remaining accumulated indices as a final partial batch
        if self._batch_accumulator:
            self._flush_batch_accumulator()

        if not complete:
            logger.info(
                "Epoch %d incomplete (early break / GeneratorExit) for %s loader; "
                "skipping end-of-epoch checks.",
                self.epoch,
                self.role,
            )
            return

        if self.completeness is not None:
            vs = self.completeness.end_epoch()
            self.violations.extend(vs)

        if self.shuffle is not None:
            vs = self.shuffle.end_epoch()
            self.violations.extend(vs)

        if self.sequential_order is not None:
            vs = self.sequential_order.end_epoch()
            self.violations.extend(vs)

        self._check_transform_stats()

    def _check_transform_stats(self) -> None:
        if self.monitored_compose is None:
            return
        import math
        for i, sr in enumerate(self.monitored_compose.get_step_reports()):
            if sr.get("dtype_truncations"):
                trunc = sr["dtype_truncations"][0]
                self.violations.append(Violation(
                    type=ViolationType.DTYPE_TRUNCATION,
                    message=f"Transform step {i} ({sr['name']}): dtype truncated "
                            f"from {trunc['from']} to {trunc['to']}",
                    epoch=self.epoch,
                    details={"step": i, "name": sr["name"],
                             "from_dtype": trunc["from"],
                             "to_dtype": trunc["to"]},
                ))
            total = sr["total_count"]
            if total == 0:
                if (self._getitem_count > 0
                        and i not in self._transform_never_invoked_fired):
                    self._transform_never_invoked_fired.add(i)
                    self.violations.append(Violation(
                        type=ViolationType.TRANSFORM_NEVER_INVOKED,
                        message=f"Transform step {i} ({sr['name']}): present in "
                                f"pipeline but never invoked across "
                                f"{self._getitem_count} samples; transform is "
                                f"being bypassed",
                        epoch=self.epoch,
                        details={"step": i, "name": sr["name"],
                                 "getitem_count": self._getitem_count},
                    ))
                continue
            if i in self._transform_noop_fired:
                continue
            unchecked = sr["unchecked_count"]
            checked = total - unchecked
            noop = sr["noop_count"]
            if checked <= 0 or noop < checked:
                continue
            p_attr = sr.get("p")
            if p_attr is not None and 0 < p_attr < 1:
                n_min = math.ceil(math.log(1e-6) / math.log(1 - p_attr))
                if checked < n_min:
                    continue  # UNDETERMINED: too few checked calls
            elif checked < CHECK_WINDOW:
                continue  # too few samples to call a no-op
            self._transform_noop_fired.add(i)
            self.violations.append(Violation(
                type=ViolationType.TRANSFORM_NOOP,
                message=f"Transform step {i} ({sr['name']}): all {checked} "
                        f"checked calls were no-ops; transform has no effect",
                epoch=self.epoch,
                details={"step": i, "name": sr["name"], "checked": checked},
            ))

    def on_getitem(self, idx: int, raw: object) -> None:
        self._getitem_count += 1
        if (not self.label_check_enabled or self.label_cache is None
                or idx in self._label_mismatch_fired):
            return
        if not isinstance(raw, (tuple, list)) or len(raw) < 2:
            return
        try:
            label = int(raw[1])
        except (TypeError, ValueError):
            return
        if 0 <= idx < len(self.label_cache) and label != self.label_cache[idx]:
            self._label_mismatch_fired.add(idx)
            self.violations.append(Violation(
                type=ViolationType.LABEL_MISMATCH,
                message=f"Sample {idx}: __getitem__ label {label} != cached "
                        f"label {self.label_cache[idx]}; labels shifted "
                        f"since guard() time",
                epoch=self.epoch,
                details={"idx": idx, "got": label,
                         "cached": self.label_cache[idx]},
            ))

    def get_violations(self) -> list[Violation]:
        return list(self.violations)
