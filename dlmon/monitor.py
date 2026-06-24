from __future__ import annotations

import logging
import math
import warnings
from typing import Any

log = logging.getLogger("dlmon")

import torch
from torch.utils.data import DataLoader, BatchSampler, IterableDataset, Sampler

from dlmon.invariants import (
    BatchDistribution,
    PartitionExclusive,
    Violation,
    ViolationType,
)
from dlmon.sampler_guard import (
    BatchSamplerGuard,
    SamplerGuard,
    SamplerSpec,
    classify_sampler,
)
from dlmon.guard_state import GuardState
from dlmon.dataset_guard import DatasetGuard, MonitoredCompose, _equal, _snapshot


class GuardError(RuntimeError):
    pass


class GuardedLoader:
    """Iterable wrapper around a DataLoader for runtime monitoring.

    Iterate this object instead of the raw DataLoader.
    Forwards attribute access to the underlying loader for transparency.
    """

    COLLATE_CHECK_RATE = 10

    def __init__(self, loader: DataLoader, state: GuardState):
        self._loader = loader
        self._state = state
        self._batch_idx = 0
        self._shape_mismatch_fired = False
        self._collate_fired = False
        self._num_workers = getattr(loader, "num_workers", 0)

    def __iter__(self):
        self._batch_idx = 0
        for batch in self._loader:
            self._batch_idx += 1
            self._check_batch(batch)
            yield batch

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)

    def _check_batch(self, batch) -> None:
        self._check_nan_inf(batch)
        self._check_shape_dtype(batch)
        # Collate check requires sampler-index/batch-content synchrony,
        # which breaks under worker prefetching
        if self._num_workers == 0 and self._batch_idx % self.COLLATE_CHECK_RATE == 1:
            self._check_collate_labels(batch)

    def _check_nan_inf(self, batch) -> None:
        tensors = _extract_tensors(batch)
        for i, t in enumerate(tensors):
            has_nan = bool(torch.isnan(t).any().item())
            has_inf = bool(torch.isinf(t).any().item())
            if has_nan or has_inf:
                problems = []
                if has_nan:
                    problems.append("NaN")
                if has_inf:
                    problems.append("Inf")
                self._state.violations.append(Violation(
                    type=ViolationType.NAN_INF_OUTPUT,
                    message=f"Batch {self._batch_idx} element {i} contains "
                            f"{'+'.join(problems)}",
                    epoch=self._state.epoch,
                    batch_idx=self._batch_idx,
                    details={"batch_idx": self._batch_idx, "element": i,
                             "has_nan": has_nan, "has_inf": has_inf},
                ))

    def _check_shape_dtype(self, batch) -> None:
        if self._shape_mismatch_fired:
            return
        sig = _batch_signature(batch)
        if not sig:
            return
        state = self._state
        if state._batch_reference_sig is None:
            state._batch_reference_sig = sig
            return
        ref = state._batch_reference_sig
        for i, (r, c) in enumerate(zip(ref, sig)):
            if r["dtype"] != c["dtype"]:
                self._shape_mismatch_fired = True
                state.violations.append(Violation(
                    type=ViolationType.SHAPE_MISMATCH,
                    message=f"Batch {self._batch_idx} element {i}: dtype "
                            f"{c['dtype']} differs from reference {r['dtype']}",
                    epoch=state.epoch,
                    batch_idx=self._batch_idx,
                    details={"element": i, "expected_dtype": r["dtype"],
                             "actual_dtype": c["dtype"]},
                ))
                return
            r_tail = r["shape"][1:]
            c_tail = c["shape"][1:]
            if r_tail and c_tail and r_tail != c_tail:
                self._shape_mismatch_fired = True
                state.violations.append(Violation(
                    type=ViolationType.SHAPE_MISMATCH,
                    message=f"Batch {self._batch_idx} element {i}: shape "
                            f"{c['shape']} differs from reference {r['shape']} "
                            f"(non-batch dimensions changed)",
                    epoch=state.epoch,
                    batch_idx=self._batch_idx,
                    details={"element": i, "expected_shape": list(r["shape"]),
                             "actual_shape": list(c["shape"])},
                ))
                return

    def _check_collate_labels(self, batch) -> None:
        if self._collate_fired:
            return
        state = self._state
        if state.label_cache is None or not state.last_batch_indices:
            return
        expected = []
        for idx in state.last_batch_indices:
            if 0 <= idx < len(state.label_cache):
                expected.append(state.label_cache[idx])
        if not expected:
            return
        actual = _extract_labels(batch)
        if actual is None or len(actual) != len(expected):
            return
        if expected != actual:
            mismatches = sum(1 for a, b in zip(expected, actual) if a != b)
            self._collate_fired = True
            state.violations.append(Violation(
                type=ViolationType.COLLATE_LABEL_CORRUPTION,
                message=f"Batch {self._batch_idx}: {mismatches}/{len(expected)} "
                        f"labels differ from expected based on sampler indices",
                epoch=state.epoch,
                batch_idx=self._batch_idx,
                details={"mismatches": mismatches,
                         "batch_size": len(expected),
                         "expected_labels": expected[:8],
                         "actual_labels": actual[:8]},
            ))


class DLMonitor:

    def __init__(
        self,
        dataset_size: int | None = None,
        check_every_n: int = 1,
    ):
        self.dataset_size = dataset_size
        self.check_every_n = check_every_n

        self.partitions = PartitionExclusive()
        self.violations: list[Violation] = []
        self._attached = False

        self._guarded_states: dict[int, GuardState] = {}
        self._guarded_loader_ids: set[int] = set()

    def guard(
        self,
        loader: DataLoader,
        role: str = "train",
        expect_duplicates: bool = False,
        label_fn: Any = None,
        dataset_monitoring: bool = True,
        learned_bounds: dict | None = None,
        allow_stochastic_eval: bool = False,
    ) -> GuardedLoader:
        """Wrap a DataLoader for runtime monitoring.

        Returns a GuardedLoader iterable. Iterate the returned object
        instead of the original DataLoader.

        Dataset-level checks need num_workers=0; with workers > 0 only
        sampler-level and guard-time checks are active.
        """
        self._compat_self_check()

        if id(loader) in self._guarded_loader_ids:
            raise GuardError("loader is already guarded")

        dataset = loader.dataset
        num_workers = getattr(loader, "num_workers", 0)
        is_iterable = isinstance(dataset, IterableDataset)
        dataset_len = 0 if is_iterable else (len(dataset) if hasattr(dataset, "__len__") else 0)

        if is_iterable:
            if learned_bounds is not None:
                warnings.warn(
                    "DLMon: learned_bounds provided but loader uses "
                    "IterableDataset; BatchDistribution is not applicable. "
                    "Bounds ignored.",
                    stacklevel=2,
                )
            spec = SamplerSpec()
            if expect_duplicates:
                spec.uniqueness = False
            if role != "train":
                spec.shuffle_expected = False

            state = GuardState(spec=spec, dataset_len=0, role=role)

            if num_workers == 0:
                from dlmon.iterable_guard import IterableGuard
                object.__setattr__(loader, "dataset", IterableGuard(dataset, state))
                state.dataset_monitoring_status = "ITERABLE_ACTIVE"
            else:
                warnings.warn(
                    "DLMon: IterableDataset monitoring is disabled because "
                    "num_workers > 0. Content-hash duplicate detection requires "
                    "num_workers=0.",
                    stacklevel=2,
                )
                state.dataset_monitoring_status = "DISABLED_NUM_WORKERS"

            state.label_cache_status = "NOT_APPLICABLE"

            self._guarded_states[id(loader)] = state
            self._guarded_loader_ids.add(id(loader))
            return GuardedLoader(loader, state)

        inner_sampler = self._get_inner_sampler(loader)
        spec = classify_sampler(inner_sampler, dataset_len)

        if expect_duplicates:
            spec.uniqueness = False

        if role != "train":
            spec.shuffle_expected = False

        state = GuardState(spec=spec, dataset_len=dataset_len, role=role)
        state.dataset_ref = dataset

        if self._is_auto_batching_stock(loader):
            guard = SamplerGuard(loader.batch_sampler.sampler, state)
            loader.batch_sampler.sampler = guard
            state._batch_size = loader.batch_sampler.batch_size
        elif loader.batch_sampler is not None:
            guard = BatchSamplerGuard(loader.batch_sampler, state)
            object.__setattr__(loader, "batch_sampler", guard)
        else:
            guard = SamplerGuard(loader.sampler, state)
            object.__setattr__(loader, "sampler", guard)

        if not dataset_monitoring:
            state.dataset_monitoring_status = "DISABLED_BY_CONFIG"
            state.label_cache_status = "DISABLED_BY_CONFIG"
            if learned_bounds is not None:
                warnings.warn(
                    "DLMon: learned_bounds provided but dataset_monitoring "
                    "is disabled; BatchDistribution requires label cache. "
                    "Bounds ignored.",
                    stacklevel=2,
                )
        else:
            self._build_label_cache(dataset, dataset_len, state, label_fn)

            if learned_bounds is not None:
                self._inject_learned_bounds(state, learned_bounds)

            if num_workers == 0:
                self._wire_dataset_monitoring(loader, dataset, state)
                state.dataset_monitoring_status = "ACTIVE"
                state.label_check_enabled = (
                    state.label_cache is not None
                    and getattr(dataset, "target_transform", None) is None
                )
            else:
                warnings.warn(
                    "DLMon: dataset-level monitoring (NaN detection, dtype "
                    "truncation, transform no-op, label cross-check) "
                    "is DISABLED because num_workers > 0. These checks require "
                    "num_workers=0. Only sampler-level invariants + guard-time "
                    "checks are active. "
                    "Use monitor.preflight(dataset) for pre-training transform checks.",
                    stacklevel=2,
                )
                state.dataset_monitoring_status = "DISABLED_NUM_WORKERS"

        if role != "train" and not allow_stochastic_eval:
            self._check_stochastic_eval(loader, state)

        if num_workers > 0 and not is_iterable:
            self._check_worker_seeds(loader, dataset, state, num_workers)

        self._guarded_states[id(loader)] = state
        self._guarded_loader_ids.add(id(loader))

        return GuardedLoader(loader, state)

    def export_bounds(self, loader: DataLoader | GuardedLoader) -> dict:
        """Export learned BatchDistribution bounds for later re-injection."""
        if isinstance(loader, GuardedLoader):
            state = self._guarded_states.get(id(loader._loader))
        else:
            state = self._guarded_states.get(id(loader))
        if state is None:
            raise GuardError("loader is not guarded by this monitor")

        bd = state.batch_dist
        if bd is None:
            raise GuardError(
                "loader has no BatchDistribution (no label source); "
                "cannot export bounds"
            )

        if bd.learning and bd._learning_stats:
            bd._finalize_learning()
            bd.learning = False
            bd._learning_stats.clear()
        elif bd.learning and not bd._learning_stats:
            raise GuardError(
                "No batch observations recorded; calibration epoch was "
                "empty or incomplete"
            )

        return {
            "version": 1,
            "sigma_factor": bd.sigma_factor,
            "batch_size": state._batch_size,
            "bounds": {k: list(v) for k, v in bd.learned_bounds.items()},
            "metadata": {
                "dataset_len": state.dataset_len,
                "num_classes": len(set(state.label_cache)) if state.label_cache else 0,
                "learning_epoch_batches": state._batch_count,
            },
        }

    def _inject_learned_bounds(self, state: GuardState, learned_bounds: dict) -> None:

        version = learned_bounds.get("version")
        if version != 1:
            raise GuardError(
                f"learned_bounds version {version} is not supported (expected 1)"
            )

        bounds = learned_bounds.get("bounds")
        if not isinstance(bounds, dict):
            raise GuardError(
                "learned_bounds['bounds'] must be a dict, "
                f"got {type(bounds).__name__}"
            )
        for k, v in bounds.items():
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise GuardError(
                    f"learned_bounds['bounds']['{k}'] must be a 2-element "
                    f"sequence, got {v!r}"
                )

        if state.batch_dist is None:
            log.warning(
                "DLMon: learned_bounds provided but no label source; "
                "bounds ignored."
            )
            return

        state.batch_dist.learned_bounds = {
            k: tuple(v) for k, v in bounds.items()
        }
        state.batch_dist.learning = False
        state.batch_dist._learning_stats.clear()

        sf = learned_bounds.get("sigma_factor")
        if sf is not None and sf != state.batch_dist.sigma_factor:
            log.warning(
                "DLMon: learned_bounds sigma_factor=%.1f differs from "
                "current sigma_factor=%.1f; bounds are pre-computed, this "
                "is informational only.",
                sf, state.batch_dist.sigma_factor,
            )

        lb_bs = learned_bounds.get("batch_size")
        if (lb_bs is not None and state._batch_size is not None
                and lb_bs != state._batch_size):
            log.warning(
                "DLMon: bounds were learned at batch_size=%d but this "
                "loader uses batch_size=%d; per-batch class counts will "
                "differ, likely producing false positives or missed "
                "detections. Re-calibrate with the correct batch_size.",
                lb_bs, state._batch_size,
            )

        if state.label_cache is not None:
            expected_keys = {f"class_{c}_count" for c in set(state.label_cache)}
            bound_keys = set(bounds.keys())
            if expected_keys != bound_keys:
                extra = bound_keys - expected_keys
                missing = expected_keys - bound_keys
                parts = []
                if extra:
                    parts.append(f"extra in bounds: {sorted(extra)}")
                if missing:
                    parts.append(f"missing from bounds: {sorted(missing)}")
                log.warning(
                    "DLMon: class set mismatch between learned_bounds and "
                    "current dataset; %s",
                    "; ".join(parts),
                )

    _compat_checked = False

    @classmethod
    def _compat_self_check(cls) -> None:
        if cls._compat_checked:
            return
        try:
            scratch = DataLoader([(0, 0)], batch_size=1)
            _ = scratch.batch_sampler.sampler
            scratch.batch_sampler.sampler = scratch.batch_sampler.sampler
            # DataLoader.__setattr__ blocks post-init writes to public attrs;
            # verify object.__setattr__ bypass works for the ones we need
            object.__setattr__(scratch, "dataset", scratch.dataset)
        except Exception as e:
            raise GuardError(
                f"dlmon compatibility self-check failed on torch "
                f"{torch.__version__}: {e}."
            ) from e
        cls._compat_checked = True

    def _get_inner_sampler(self, loader: DataLoader) -> Sampler:
        if self._is_auto_batching_stock(loader):
            return loader.batch_sampler.sampler
        if loader.batch_sampler is not None:
            return loader.batch_sampler
        return loader.sampler

    @staticmethod
    def _is_auto_batching_stock(loader: DataLoader) -> bool:
        return type(loader.batch_sampler) is BatchSampler

    @staticmethod
    def _build_label_cache(
        dataset: Any, dataset_len: int, state: Any, label_fn: Any
    ) -> None:
        cache: list[int] | None = None

        if label_fn is not None:
            try:
                cache = [int(label_fn(dataset, i)) for i in range(dataset_len)]
            except Exception:
                cache = None

        if cache is None:
            targets = getattr(dataset, "targets", None)
            if targets is not None:
                try:
                    cache = [int(targets[i]) for i in range(dataset_len)]
                except Exception:
                    cache = None

        if cache is None:
            labels = getattr(dataset, "labels", None)
            if labels is not None:
                try:
                    cache = [int(labels[i]) for i in range(dataset_len)]
                except Exception:
                    cache = None

        if cache is not None:
            state.label_cache = cache
            state.batch_dist = BatchDistribution()
            state.label_cache_status = "ACTIVE"
        else:
            state.label_cache = None
            state.batch_dist = None
            state.label_cache_status = "DISABLED_NO_LABEL_SOURCE"
            log.info(
                "DLMon: no label source found (label_fn / .targets / .labels). "
                "BatchDistribution monitoring is disabled for this loader."
            )

    @staticmethod
    def _wire_dataset_monitoring(
        loader: DataLoader, dataset: Any, state: GuardState
    ) -> None:
        guarded_ds = DatasetGuard(dataset, state)
        monitored_t = DLMonitor._discover_and_wrap_transforms(guarded_ds)
        state.monitored_compose = monitored_t
        object.__setattr__(loader, "dataset", guarded_ds)

    @staticmethod
    def _discover_and_wrap_transforms(guarded_ds: DatasetGuard) -> MonitoredCompose | None:
        inner = guarded_ds.inner

        monitored_t = DLMonitor._wrap_if_present(inner, "transform")
        monitored_tt = DLMonitor._wrap_if_present(inner, "target_transform")

        if hasattr(inner, "transforms") and monitored_t is not None:
            try:
                from torchvision.datasets.vision import StandardTransform
                if isinstance(inner.transforms, StandardTransform):
                    inner.transforms = StandardTransform(
                        monitored_t,
                        monitored_tt,
                    )
            except ImportError:
                pass

        return monitored_t

    @staticmethod
    def _check_worker_seeds(
        loader: DataLoader, dataset: Any, state: "GuardState",
        num_workers: int,
    ) -> None:
        import xxhash

        transform = getattr(dataset, "transform", None)
        if transform is None:
            return
        steps = _extract_transform_list(transform)
        if not any(_is_claimed_stochastic(s) for s in steps):
            return

        worker_init_fn = getattr(loader, "worker_init_fn", None)
        if worker_init_fn is None:
            return

        ds_len = len(dataset) if hasattr(dataset, "__len__") else 0
        if ds_len == 0:
            return

        n_test = min(8, ds_len)
        test_indices = list(range(0, ds_len, max(1, ds_len // n_test)))[:n_test]
        n_sim = min(num_workers, 4)

        worker_hashes = []
        for w in range(n_sim):
            torch_state = torch.random.get_rng_state()
            try:
                import numpy as np
                np_state = np.random.get_state()
            except ImportError:
                np_state = None
            try:
                import random as _random
                py_state = _random.getstate()
            except Exception:
                py_state = None

            try:
                worker_init_fn(w)
                h = xxhash.xxh64()
                for idx in test_indices:
                    item = dataset[idx]
                    x = item[0] if isinstance(item, (tuple, list)) else item
                    if isinstance(x, torch.Tensor):
                        h.update(x.detach().cpu().numpy().tobytes())
                worker_hashes.append(h.intdigest())
            finally:
                torch.random.set_rng_state(torch_state)
                if np_state is not None:
                    try:
                        import numpy as np
                        np.random.set_state(np_state)
                    except Exception:
                        pass
                if py_state is not None:
                    try:
                        import random as _random
                        _random.setstate(py_state)
                    except Exception:
                        pass

        if len(set(worker_hashes)) == 1 and len(worker_hashes) >= 2:
            state.violations.append(Violation(
                type=ViolationType.FROZEN_WORKER_SEEDS,
                message=f"All {n_sim} simulated workers produced identical "
                        f"stochastic transform outputs; worker_init_fn may "
                        f"freeze all workers to the same RNG state",
                epoch=-1,
                details={"n_workers_simulated": n_sim,
                         "n_test_indices": len(test_indices)},
            ))

    @staticmethod
    def _check_stochastic_eval(loader: DataLoader, state: "GuardState") -> None:
        dataset = loader.dataset
        inner = dataset.inner if isinstance(dataset, DatasetGuard) else dataset
        for attr in ("transform", "target_transform"):
            fn = getattr(inner, attr, None)
            if fn is None:
                continue
            steps = _extract_transform_list(fn)
            for step in steps:
                if _is_claimed_stochastic(step):
                    state.violations.append(Violation(
                        type=ViolationType.STOCHASTIC_EVAL_TRANSFORM,
                        message=f"Stochastic transform {type(step).__name__} on "
                                f"'{state.role}' loader; evaluation will be "
                                f"non-deterministic",
                        epoch=-1,
                        details={"transform": type(step).__name__,
                                 "role": state.role, "attr": attr},
                    ))
                    return

    @staticmethod
    def _wrap_if_present(
        dataset: Any, attr: str, always_monitor: bool = False
    ) -> MonitoredCompose | None:
        fn = getattr(dataset, attr, None)
        if fn is None:
            return None

        if isinstance(fn, MonitoredCompose):
            return fn

        steps = _extract_transform_list(fn)
        monitored = MonitoredCompose(steps, always_monitor=always_monitor)
        setattr(dataset, attr, monitored)
        return monitored

    @staticmethod
    def preflight(
        dataset: Any,
        n: int = 256,
        repeats: int = 4,
        expect_stochastic: bool | None = None,
    ) -> dict:
        """Run transform checks in the main process before training starts."""
        ds_len = len(dataset)
        if ds_len == 0:
            return {"status": "EMPTY_DATASET", "violations": []}

        stride = max(1, ds_len // n)
        indices = [i * stride for i in range(min(n, ds_len))]

        orig_transform = getattr(dataset, "transform", None)
        orig_target_transform = getattr(dataset, "target_transform", None)
        orig_transforms_plural = getattr(dataset, "transforms", None)

        monitored_t = DLMonitor._wrap_if_present(dataset, "transform", always_monitor=True)
        monitored_tt = DLMonitor._wrap_if_present(dataset, "target_transform", always_monitor=True)

        if hasattr(dataset, "transforms") and monitored_t is not None:
            try:
                from torchvision.datasets.vision import StandardTransform
                if isinstance(orig_transforms_plural, StandardTransform):
                    dataset.transforms = StandardTransform(monitored_t, monitored_tt)
            except ImportError:
                pass

        if expect_stochastic is not None:
            stochastic_pipeline = expect_stochastic
        else:
            stochastic_pipeline = monitored_t is not None and any(
                _is_claimed_stochastic(t) for t in monitored_t.transforms
            )

        violations: list[dict] = []
        shape_log: dict[int, list[tuple]] = {}

        try:
            for idx in indices:
                outputs = []
                for _ in range(repeats):
                    item = dataset[idx]
                    x = item[0] if isinstance(item, (tuple, list)) else item
                    outputs.append(x)

                    shape_dtype = _get_shape_dtype(x)
                    shape_log.setdefault(idx, []).append(shape_dtype)

                shapes = shape_log[idx]
                if shapes and len(set(shapes)) > 1:
                    violations.append({
                        "type": ViolationType.SHAPE_MISMATCH.name,
                        "idx": idx,
                        "shapes": [str(s) for s in shapes],
                    })

                if stochastic_pipeline and repeats >= 2 and _all_identical(outputs):
                    violations.append({
                        "type": "FROZEN_STOCHASTIC",
                        "idx": idx,
                        "message": f"All {repeats} repeats of index {idx} produced "
                                   f"identical output; randomness may be frozen",
                    })
        finally:
            if orig_transform is not None:
                dataset.transform = orig_transform
            elif hasattr(dataset, "transform") and orig_transform is None:
                try:
                    del dataset.transform
                except (AttributeError, TypeError):
                    pass

            if orig_target_transform is not None:
                dataset.target_transform = orig_target_transform
            elif hasattr(dataset, "target_transform") and orig_target_transform is None:
                try:
                    del dataset.target_transform
                except (AttributeError, TypeError):
                    pass

            if orig_transforms_plural is not None:
                dataset.transforms = orig_transforms_plural

        step_reports = []
        if monitored_t is not None:
            step_reports = monitored_t.get_step_reports()
            for i, sr in enumerate(step_reports):
                total = sr["total_count"]
                if total == 0:
                    continue
                noop = sr["noop_count"]
                unchecked = sr["unchecked_count"]
                p_attr = sr.get("p")

                status = "OK"
                if unchecked == total:
                    status = "UNCHECKED"
                elif noop == total - unchecked and noop > 0:
                    checked = total - unchecked
                    if p_attr is not None and 0 < p_attr < 1:
                        n_min = math.ceil(math.log(1e-6) / math.log(1 - p_attr))
                        if checked < n_min:
                            status = "UNDETERMINED"
                        else:
                            status = "VIOLATION"
                            violations.append({
                                "type": "TRANSFORM_NOOP",
                                "step": i,
                                "name": sr["name"],
                                "noop_rate": noop / total,
                            })
                    else:
                        status = "VIOLATION"
                        violations.append({
                            "type": "TRANSFORM_NOOP",
                            "step": i,
                            "name": sr["name"],
                            "noop_rate": noop / total,
                        })
                sr["status"] = status

        return {
            "status": "COMPLETE",
            "indices_sampled": len(indices),
            "repeats": repeats,
            "frozen_check": ("ENABLED" if stochastic_pipeline
                             else "SKIPPED_DETERMINISTIC_PIPELINE"),
            "step_reports": step_reports,
            "violations": violations,
        }

    def attach(self, loader: DataLoader | None = None) -> GuardedLoader:
        if loader is None:
            raise GuardError(
                "attach() with no argument (global DataLoader patching) is removed. "
                "Use monitor.guard(loader) for each DataLoader instead."
            )
        warnings.warn(
            "DLMonitor.attach(loader) is deprecated. "
            "Use: guarded = monitor.guard(loader, role=...) and iterate "
            "'guarded' instead of the raw loader.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.guard(loader)

    def detach(self) -> None:
        self._attached = False

    def on_epoch_start(self, epoch: int | None = None) -> None:
        pass

    def on_epoch_end(self) -> list[Violation]:
        return []

    def register_partition(self, name: str, sample_ids) -> None:
        self.partitions.register(name, sample_ids)

    def check_partitions(self) -> list[Violation]:
        vs = self.partitions.check()

        states = list(self._guarded_states.values())
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                a, b = states[i], states[j]
                if (a.dataset_ref is None or a.dataset_ref is not b.dataset_ref
                        or a.role == b.role):
                    continue
                overlap = a.all_seen_indices & b.all_seen_indices
                if overlap:
                    v = Violation(
                        type=ViolationType.PARTITION_OVERLAP,
                        message=f"{len(overlap)} sample indices observed by both "
                                f"'{a.role}' and '{b.role}' loaders over the same "
                                f"dataset object",
                        epoch=-1,
                        details={"roles": (a.role, b.role),
                                 "overlap_count": len(overlap),
                                 "examples": sorted(overlap)[:10]},
                    )
                    vs.append(v)

        self.violations.extend(vs)
        return vs

    def status(self) -> dict:
        result: dict[int, dict] = {}
        for loader_id, state in self._guarded_states.items():
            spec = state.spec
            result[loader_id] = {
                "role": state.role,
                "epochs_completed": state.epoch,
                "completeness_mode": spec.completeness.name,
                "uniqueness": {
                    "active": bool(spec.uniqueness),
                    "dup_tolerance": spec.dup_tolerance,
                },
                "shuffle_detector_active": bool(
                    spec.shuffle_expected and state.role == "train"
                ),
                "dataset_monitoring_status": state.dataset_monitoring_status,
                "label_cache_status": state.label_cache_status,
            }
        return result

    def summary(self) -> str:
        all_violations: list[Violation] = []
        lines = ["DLMon summary:"]
        lines.append(f"  Guarded loaders: {len(self._guarded_states)}")

        status = self.status()
        for loader_id, state in self._guarded_states.items():
            st = status[loader_id]
            n_v = len(state.violations)
            uq = st["uniqueness"]
            lines.append(
                f"  [{st['role']}] epochs={st['epochs_completed']}, violations={n_v}, "
                f"dataset_monitoring={st['dataset_monitoring_status']}, "
                f"label_cache={st['label_cache_status']}, "
                f"completeness={st['completeness_mode']}, "
                f"uniqueness={'ON' if uq['active'] else 'OFF'}"
                f"(dup_tolerance={uq['dup_tolerance']}), "
                f"shuffle_detector={'ON' if st['shuffle_detector_active'] else 'OFF'}"
            )
            all_violations.extend(state.violations)

        all_violations.extend(self.violations)

        by_type: dict[str, int] = {}
        for v in all_violations:
            by_type[v.type.name] = by_type.get(v.type.name, 0) + 1

        lines.append(f"  Total violations: {len(all_violations)}")
        for vtype, count in sorted(by_type.items()):
            lines.append(f"    {vtype}: {count}")

        return "\n".join(lines)

    def get_violations(self) -> list[Violation]:
        result: list[Violation] = []
        for state in self._guarded_states.values():
            result.extend(state.violations)
        result.extend(self.violations)
        return result



def _extract_tensors(batch: Any) -> list[torch.Tensor]:
    if isinstance(batch, torch.Tensor):
        return [batch]
    if isinstance(batch, dict):
        return [v for v in batch.values() if isinstance(v, torch.Tensor)]
    if isinstance(batch, (tuple, list)):
        return [v for v in batch if isinstance(v, torch.Tensor)]
    return []


def _batch_signature(batch: Any) -> list[dict]:
    tensors = _extract_tensors(batch)
    return [{"shape": tuple(t.shape), "dtype": str(t.dtype)} for t in tensors]


def _extract_labels(batch: Any) -> list[int] | None:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        labels = batch[1]
        if isinstance(labels, torch.Tensor):
            try:
                return labels.tolist()
            except Exception:
                return None
    if isinstance(batch, dict):
        for key in ("label", "labels", "target", "targets", "y"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                try:
                    return batch[key].tolist()
                except Exception:
                    return None
    return None


def _extract_transform_list(fn: Any) -> list:
    if hasattr(fn, "transforms") and isinstance(fn.transforms, list):
        return fn.transforms
    return [fn]


_KNOWN_STOCHASTIC_TRANSFORMS = {
    "ColorJitter", "AutoAugment", "RandAugment", "TrivialAugmentWide",
    "AugMix", "GaussianBlur", "ElasticTransform", "GaussianNoise",
}


def _is_claimed_stochastic(t: Any) -> bool:
    p = getattr(t, "p", None)
    if isinstance(p, (int, float)) and 0 < p < 1:
        return True
    name = type(t).__name__
    return name.startswith("Random") or name in _KNOWN_STOCHASTIC_TRANSFORMS


def _get_shape_dtype(x: Any) -> tuple:
    if isinstance(x, torch.Tensor):
        return (tuple(x.shape), str(x.dtype))
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return (x.shape, str(x.dtype))
    except ImportError:
        pass
    try:
        from PIL import Image as PILImage
        if isinstance(x, PILImage.Image):
            return (x.size, x.mode)
    except ImportError:
        pass
    return (type(x).__name__,)


def _all_identical(outputs: list) -> bool:
    if len(outputs) < 2:
        return False
    first = outputs[0]
    for other in outputs[1:]:
        eq, method = _equal(first, other)
        if method == "UNCHECKED" or not eq:
            return False
    return True
