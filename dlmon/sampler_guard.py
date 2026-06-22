from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from torch.utils.data import Sampler
from torch.utils.data import (
    SequentialSampler,
    RandomSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

from dlmon.invariants import CompletenessMode

if TYPE_CHECKING:
    from dlmon.guard_state import GuardState


@dataclass
class SamplerSpec:
    uniqueness: bool = True
    dup_tolerance: int = 0
    completeness: CompletenessMode = CompletenessMode.OFF
    expected_set: set[int] | None = None
    expected_count: int | None = None
    shuffle_expected: bool = False
    sequential_provenance: str = "unknown"


def classify_sampler(sampler: Sampler, dataset_len: int) -> SamplerSpec:
    if type(sampler) is SequentialSampler:
        return SamplerSpec(
            uniqueness=True,
            dup_tolerance=0,
            completeness=CompletenessMode.EXACT_SET,
            expected_set=set(range(dataset_len)),
            shuffle_expected=False,
            sequential_provenance="structural",
        )

    if type(sampler) is RandomSampler:
        replacement = getattr(sampler, "replacement", False)
        if replacement:
            return SamplerSpec(
                uniqueness=False,
                completeness=CompletenessMode.OFF,
                shuffle_expected=True,
            )
        num_samples = getattr(sampler, "num_samples", dataset_len)
        if num_samples == dataset_len:
            return SamplerSpec(
                uniqueness=True,
                dup_tolerance=0,
                completeness=CompletenessMode.EXACT_SET,
                expected_set=set(range(dataset_len)),
                shuffle_expected=True,
            )
        return SamplerSpec(
            uniqueness=True,
            dup_tolerance=0,
            completeness=CompletenessMode.COUNT_ONLY,
            expected_count=num_samples,
            shuffle_expected=True,
        )

    if type(sampler) is SubsetRandomSampler:
        indices = getattr(sampler, "indices", None)
        if indices is not None:
            return SamplerSpec(
                uniqueness=True,
                dup_tolerance=0,
                completeness=CompletenessMode.EXACT_SET,
                expected_set=set(indices),
                shuffle_expected=True,
            )
        return SamplerSpec(uniqueness=True, shuffle_expected=True)

    if type(sampler) is WeightedRandomSampler:
        replacement = getattr(sampler, "replacement", True)
        if replacement:
            return SamplerSpec(
                uniqueness=False,
                completeness=CompletenessMode.OFF,
                shuffle_expected=True,
            )
        num_samples = getattr(sampler, "num_samples", dataset_len)
        return SamplerSpec(
            uniqueness=True,
            dup_tolerance=0,
            completeness=CompletenessMode.COUNT_ONLY,
            expected_count=num_samples,
            shuffle_expected=True,
        )

    if type(sampler) is DistributedSampler:
        drop_last = getattr(sampler, "drop_last", False)
        num_replicas = getattr(sampler, "num_replicas", 1)
        total_size = getattr(sampler, "total_size", dataset_len)
        ds_len = len(getattr(sampler, "dataset", [])) if hasattr(sampler, "dataset") else dataset_len
        num_samples = getattr(sampler, "num_samples", dataset_len)
        shuffle = getattr(sampler, "shuffle", True)

        tol = _distributed_dup_tolerance(total_size, ds_len, num_replicas, drop_last)

        return SamplerSpec(
            uniqueness=True,
            dup_tolerance=tol,
            completeness=CompletenessMode.COUNT_ONLY,
            expected_count=num_samples,
            shuffle_expected=shuffle,
        )

    # duck-typing fallback for subclasses
    if _has_attrs(sampler, "replacement", "num_samples"):
        replacement = getattr(sampler, "replacement", False)
        num_samples = getattr(sampler, "num_samples", dataset_len)
        if replacement:
            return SamplerSpec(uniqueness=False, completeness=CompletenessMode.OFF, shuffle_expected=True)
        if num_samples == dataset_len:
            return SamplerSpec(
                uniqueness=True, completeness=CompletenessMode.EXACT_SET,
                expected_set=set(range(dataset_len)), shuffle_expected=True,
            )
        return SamplerSpec(
            uniqueness=True, completeness=CompletenessMode.COUNT_ONLY,
            expected_count=num_samples, shuffle_expected=True,
        )

    if _has_attrs(sampler, "indices"):
        indices = getattr(sampler, "indices", None)
        if indices is not None:
            return SamplerSpec(
                uniqueness=True, completeness=CompletenessMode.EXACT_SET,
                expected_set=set(indices), shuffle_expected=True,
            )

    if _has_attrs(sampler, "total_size", "num_replicas"):
        drop_last = getattr(sampler, "drop_last", False)
        num_replicas = getattr(sampler, "num_replicas", 1)
        total_size = getattr(sampler, "total_size", dataset_len)
        ds_len = len(getattr(sampler, "dataset", [])) if hasattr(sampler, "dataset") else dataset_len
        num_samples = getattr(sampler, "num_samples", dataset_len)
        shuffle = getattr(sampler, "shuffle", True)
        tol = _distributed_dup_tolerance(total_size, ds_len, num_replicas, drop_last)
        return SamplerSpec(
            uniqueness=True, dup_tolerance=tol,
            completeness=CompletenessMode.COUNT_ONLY,
            expected_count=num_samples, shuffle_expected=shuffle,
        )

    return SamplerSpec(
        uniqueness=True,
        dup_tolerance=0,
        completeness=CompletenessMode.OFF,
        shuffle_expected=False,
    )


def _distributed_dup_tolerance(total_size: int, ds_len: int, num_replicas: int, drop_last: bool) -> int:
    if drop_last:
        return 0
    padding = total_size - ds_len
    return math.ceil(padding / num_replicas) if padding > 0 else 0


def _has_attrs(obj: object, *names: str) -> bool:
    return all(hasattr(obj, n) for n in names)


class SamplerGuard(Sampler):

    def __init__(self, inner: Sampler, state: GuardState):
        super().__init__()
        self.inner = inner
        self.state = state

    def __len__(self) -> int:
        return len(self.inner)

    def __iter__(self) -> Iterator[int]:
        self.state.on_epoch_boundary()
        it = iter(self.inner)
        exhausted = False
        try:
            for idx in it:
                self.state.on_index(int(idx))
                yield idx
            exhausted = True
        finally:
            self.state.on_epoch_close(complete=exhausted)


class BatchSamplerGuard:

    def __init__(self, inner, state: GuardState):
        self.inner = inner
        self.state = state

    def __len__(self) -> int:
        return len(self.inner)

    def __iter__(self) -> Iterator[list[int]]:
        self.state.on_epoch_boundary()
        it = iter(self.inner)
        exhausted = False
        try:
            for batch_indices in it:
                for i in batch_indices:
                    self.state.on_index(int(i))
                self.state.on_batch_boundary(batch_indices)
                yield batch_indices
            exhausted = True
        finally:
            self.state.on_epoch_close(complete=exhausted)
