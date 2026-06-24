import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, Dataset, IterableDataset,
    RandomSampler, Sampler, SubsetRandomSampler,
    TensorDataset, WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

from dlmon import DLMonitor
from dlmon.invariants import (
    BatchDistribution, EpochCompleteness, CompletenessMode,
    SequentialOrderDetector, StochasticDiversity, TransformApplied,
    ViolationType,
)


class SimpleDataset(Dataset):
    def __init__(self, n=200, dim=10, num_classes=3):
        self.x = torch.randn(n, dim)
        self.targets = [i % num_classes for i in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


class DuplicateSampler(Sampler):
    def __init__(self, n, num_dupes=15):
        self.n = n
        self.num_dupes = num_dupes

    def __iter__(self):
        indices = list(range(self.n))
        indices[:self.num_dupes] = indices[:self.num_dupes] + indices[:self.num_dupes]
        return iter(indices[:self.n])

    def __len__(self):
        return self.n


class FixedOrderSampler(Sampler):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


def test_detect_duplicate_samples():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=16, sampler=DuplicateSampler(100, num_dupes=10))
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    dupes = [v for v in monitor.get_violations() if v.type == ViolationType.DUPLICATE_SAMPLE]
    assert len(dupes) >= 5


def test_completeness_catches_missing_samples():
    comp = EpochCompleteness(dataset_size=100, mode=CompletenessMode.EXACT_SET,
                             expected_set=set(range(100)))
    comp.start_epoch(1)
    for i in range(0, 100, 2):
        comp.record_sample(i)
    vs = comp.end_epoch()
    missing = [v for v in vs if v.type == ViolationType.MISSING_SAMPLE]
    assert len(missing) == 1
    assert missing[0].details["missing_count"] == 50


def test_detect_partition_overlap():
    monitor = DLMonitor()
    monitor.register_partition("train", list(range(0, 80)))
    monitor.register_partition("val", list(range(70, 100)))
    leaks = [v for v in monitor.check_partitions() if v.type == ViolationType.PARTITION_OVERLAP]
    assert len(leaks) == 1
    assert leaks[0].details["overlap_count"] == 10


def test_detect_no_shuffle():
    ds = SimpleDataset(100)
    gen = torch.Generator()
    loader = DataLoader(ds, batch_size=16, shuffle=True, generator=gen)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for epoch in range(3):
        gen.manual_seed(1234)
        for batch in guarded:
            pass
    bugs = [v for v in monitor.get_violations() if v.type == ViolationType.NO_SHUFFLE]
    assert len(bugs) >= 1


def test_no_false_shuffle_on_custom_sampler():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=16, sampler=FixedOrderSampler(100))
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for _ in range(3):
        for batch in guarded:
            pass
    assert len([v for v in monitor.get_violations() if v.type == ViolationType.NO_SHUFFLE]) == 0


def test_detect_transform_noop():
    checker = TransformApplied(noop_threshold=0.9)
    checker.start_epoch(0)
    for _ in range(100):
        data = torch.randn(3, 32, 32)
        checker.check("identity", data, data)
    noops = [v for v in checker.end_epoch() if v.type == ViolationType.TRANSFORM_NOOP]
    assert len(noops) == 1


def test_detect_frozen_stochastic():
    checker = StochasticDiversity(min_unique_ratio=0.1)
    checker.start_epoch(0)
    fixed = torch.randn(3, 32, 32)
    for _ in range(50):
        checker.record_output(fixed)
    frozen = [v for v in checker.end_epoch() if v.type == ViolationType.FROZEN_STOCHASTIC]
    assert len(frozen) == 1


def test_no_false_positive_on_diverse_outputs():
    checker = StochasticDiversity(min_unique_ratio=0.1)
    checker.start_epoch(0)
    for _ in range(50):
        checker.record_output(torch.randn(3, 32, 32))
    assert len([v for v in checker.end_epoch() if v.type == ViolationType.FROZEN_STOCHASTIC]) == 0


def test_batch_distribution_anomaly():
    checker = BatchDistribution()
    checker.start_epoch(0)
    for i in range(20):
        checker.observe("class_0", 48.0 + 4.0 * (i % 3), i)
    checker.start_epoch(1)
    assert len(checker.observe("class_0", 50.0, 0)) == 0
    assert len(checker.observe("class_0", 95.0, 1)) >= 1


def test_integration_clean_pipeline():
    torch.manual_seed(42)
    ds = SimpleDataset(200, dim=10, num_classes=2)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    model = nn.Linear(10, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for epoch in range(3):
        for x, y in guarded:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    assert len(monitor.get_violations()) == 0, "false positives on clean pipeline"


def test_summary():
    ds = SimpleDataset(200)
    loader = DataLoader(ds, batch_size=16, sampler=RandomSampler(ds))
    monitor = DLMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monitor.guard(loader)
    s = monitor.summary()
    assert "DLMon summary" in s
    assert "dataset_monitoring=" in s


def test_status_fields():
    loader = DataLoader(SimpleDataset(64), batch_size=16)
    monitor = DLMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monitor.guard(loader)
    st = list(monitor.status().values())[0]
    assert "role" in st
    assert "completeness_mode" in st
    assert "shuffle_detector_active" in st
    assert "dataset_monitoring_status" in st


def test_two_loaders():
    ds = SimpleDataset(200)
    train = DataLoader(ds, batch_size=16, sampler=RandomSampler(ds))
    val = DataLoader(ds, batch_size=16, shuffle=False)
    monitor = DLMonitor()
    monitor.guard(train, role="train")
    monitor.guard(val, role="val")
    status = monitor.status()
    assert len(status) == 2
    roles = {st["role"] for st in status.values()}
    assert roles == {"train", "val"}


def test_drop_last_no_false_positive():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    missing = [v for v in monitor.get_violations() if v.type == ViolationType.MISSING_SAMPLE]
    assert len(missing) == 0


# --- Integration tests for real-world loader configurations ---


class NaNDataset(Dataset):
    def __init__(self, n=100, nan_idx=50):
        self.x = torch.randn(n, 10)
        self.x[nan_idx, 0] = float("nan")
        self.targets = [i % 3 for i in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def test_num_workers_nonzero_nan_detection():
    ds = NaNDataset(100, nan_idx=50)
    loader = DataLoader(ds, batch_size=16, num_workers=2, shuffle=False)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    nans = [v for v in monitor.get_violations() if v.type == ViolationType.NAN_INF_OUTPUT]
    assert len(nans) >= 1


def test_num_workers_nonzero_no_false_positive():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=16, num_workers=2, shuffle=True)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    assert len(monitor.get_violations()) == 0


def test_distributed_sampler():
    ds = SimpleDataset(100)
    sampler = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=False)
    loader = DataLoader(ds, batch_size=16, sampler=sampler)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    sampler.set_epoch(0)
    for batch in guarded:
        pass
    violations = monitor.get_violations()
    dupes = [v for v in violations if v.type == ViolationType.DUPLICATE_SAMPLE]
    assert len(dupes) == 0


class CountingIterableDataset(IterableDataset):
    def __init__(self, n=200, dup_rate=0.0):
        self.n = n
        self.dup_rate = dup_rate

    def __iter__(self):
        for i in range(self.n):
            yield torch.tensor([float(i)]), 0
            if self.dup_rate > 0 and i % int(1.0 / self.dup_rate) == 0:
                yield torch.tensor([float(i)]), 0


def test_iterable_dataset_monitoring():
    ds = CountingIterableDataset(n=2000, dup_rate=0.1)
    loader = DataLoader(ds, batch_size=32)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    dupes = [v for v in monitor.get_violations() if v.type == ViolationType.DUPLICATE_SAMPLE]
    assert len(dupes) >= 1


def test_iterable_dataset_clean():
    ds = CountingIterableDataset(n=2000, dup_rate=0.0)
    loader = DataLoader(ds, batch_size=32)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    dupes = [v for v in monitor.get_violations() if v.type == ViolationType.DUPLICATE_SAMPLE]
    assert len(dupes) == 0


def test_persistent_workers():
    ds = SimpleDataset(200)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2,
                        persistent_workers=True)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for epoch in range(3):
        for batch in guarded:
            pass
    violations = monitor.get_violations()
    assert len(violations) == 0, f"false positives with persistent_workers: {violations}"


class DictDataset(Dataset):
    def __init__(self, n=100):
        self.x = torch.randn(n, 10)
        self.y = torch.randint(0, 3, (n,))

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return {"image": self.x[i], "label": self.y[i]}


def test_dict_batch_output():
    ds = DictDataset(100)
    loader = DataLoader(ds, batch_size=16)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train", dataset_monitoring=False)
    for batch in guarded:
        assert "image" in batch
        assert "label" in batch
    assert len(monitor.get_violations()) == 0


class MultiLabelDataset(Dataset):
    def __init__(self, n=100):
        self.x = torch.randn(n, 10)
        self.targets = [i % 3 for i in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.targets[i], i


def test_multi_label_output():
    ds = MultiLabelDataset(100)
    loader = DataLoader(ds, batch_size=16)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        assert len(batch) == 3
    assert len(monitor.get_violations()) == 0


def test_custom_collate_corruption():
    ds = SimpleDataset(100)

    def corrupt_collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.tensor([b[1] for b in batch])
        return xs, ys.flip(0)

    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=corrupt_collate)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    bugs = [v for v in monitor.get_violations() if v.type == ViolationType.COLLATE_LABEL_CORRUPTION]
    assert len(bugs) >= 1


def test_custom_collate_clean():
    ds = SimpleDataset(100)

    def clean_collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.tensor([b[1] for b in batch])
        return xs, ys

    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=clean_collate)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    bugs = [v for v in monitor.get_violations() if v.type == ViolationType.COLLATE_LABEL_CORRUPTION]
    assert len(bugs) == 0


def test_weighted_random_sampler():
    ds = SimpleDataset(100)
    weights = [1.0] * 100
    sampler = WeightedRandomSampler(weights, num_samples=100, replacement=True)
    loader = DataLoader(ds, batch_size=16, sampler=sampler)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    dupes = [v for v in monitor.get_violations() if v.type == ViolationType.DUPLICATE_SAMPLE]
    assert len(dupes) == 0


def test_subset_random_sampler():
    ds = SimpleDataset(200)
    indices = list(range(50, 150))
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(ds, batch_size=16, sampler=sampler)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    missing = [v for v in monitor.get_violations() if v.type == ViolationType.MISSING_SAMPLE]
    assert len(missing) == 0


# --- ViolationType coverage tests ---


def test_label_mismatch_detection():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=16, num_workers=0, shuffle=False)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for i in range(len(ds.targets)):
        ds.targets[i] = (ds.targets[i] + 1) % 3
    for batch in guarded:
        pass
    mismatch = [v for v in monitor.get_violations()
                if v.type == ViolationType.LABEL_MISMATCH]
    assert len(mismatch) >= 1


def test_label_mismatch_no_false_positive():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=16, num_workers=0, shuffle=False)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    mismatch = [v for v in monitor.get_violations()
                if v.type == ViolationType.LABEL_MISMATCH]
    assert len(mismatch) == 0


class BypassedTransformDataset(Dataset):
    def __init__(self, n=100):
        self.x = torch.randn(n, 10)
        self.targets = [i % 3 for i in range(n)]
        self.transform = lambda x: x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def test_transform_never_invoked_detection():
    ds = BypassedTransformDataset(100)
    loader = DataLoader(ds, batch_size=16, num_workers=0, shuffle=False)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    never = [v for v in monitor.get_violations()
             if v.type == ViolationType.TRANSFORM_NEVER_INVOKED]
    assert len(never) >= 1


class AppliedTransformDataset(Dataset):
    def __init__(self, n=100):
        self.x = torch.randn(n, 10)
        self.targets = [i % 3 for i in range(n)]
        self.transform = lambda x: x * 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.targets[i]


def test_transform_never_invoked_negative():
    ds = AppliedTransformDataset(100)
    loader = DataLoader(ds, batch_size=16, num_workers=0, shuffle=False)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    never = [v for v in monitor.get_violations()
             if v.type == ViolationType.TRANSFORM_NEVER_INVOKED]
    assert len(never) == 0


def test_unexpected_shuffle_detection():
    det = SequentialOrderDetector()
    det.start_epoch(0)
    det.record_samples(list(range(64)))
    det.end_epoch()
    det.start_epoch(1)
    det.record_samples(list(range(63, -1, -1)))
    vs = det.end_epoch()
    assert len([v for v in vs
                if v.type == ViolationType.UNEXPECTED_SHUFFLE]) == 1


def test_unexpected_shuffle_negative():
    det = SequentialOrderDetector()
    det.start_epoch(0)
    det.record_samples(list(range(64)))
    det.end_epoch()
    det.start_epoch(1)
    det.record_samples(list(range(64)))
    vs = det.end_epoch()
    assert len([v for v in vs
                if v.type == ViolationType.UNEXPECTED_SHUFFLE]) == 0


class VariableShapeDataset(Dataset):
    def __init__(self, n=32):
        self.n = n
        self.targets = [0] * n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i < self.n // 2:
            return torch.randn(10), self.targets[i]
        return torch.randn(20), self.targets[i]


def test_shape_mismatch_detection():
    ds = VariableShapeDataset(32)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    shape = [v for v in monitor.get_violations()
             if v.type == ViolationType.SHAPE_MISMATCH]
    assert len(shape) >= 1


class RandomNoiseTransform:
    p = 0.5

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return x + torch.randn_like(x) * 0.01
        return x


def test_stochastic_eval_transform_detection():
    ds = SimpleDataset(100)
    ds.transform = RandomNoiseTransform()
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    monitor = DLMonitor()
    monitor.guard(loader, role="val")
    stochastic = [v for v in monitor.get_violations()
                  if v.type == ViolationType.STOCHASTIC_EVAL_TRANSFORM]
    assert len(stochastic) >= 1


class DeterministicTransform:
    def __call__(self, x):
        return x * 2


def test_stochastic_eval_transform_negative():
    ds = SimpleDataset(100)
    ds.transform = DeterministicTransform()
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    monitor = DLMonitor()
    monitor.guard(loader, role="val")
    stochastic = [v for v in monitor.get_violations()
                  if v.type == ViolationType.STOCHASTIC_EVAL_TRANSFORM]
    assert len(stochastic) == 0


class Float64ToFloat32:
    def __call__(self, x):
        return x.float()


class Float64WithTransformDataset(Dataset):
    def __init__(self, n=100):
        self.x = torch.randn(n, 10, dtype=torch.float64)
        self.targets = [i % 3 for i in range(n)]
        self.transform = Float64ToFloat32()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.targets[i]


def test_dtype_truncation_detection():
    ds = Float64WithTransformDataset(100)
    loader = DataLoader(ds, batch_size=16, num_workers=0, shuffle=False)
    monitor = DLMonitor()
    guarded = monitor.guard(loader, role="train")
    for batch in guarded:
        pass
    trunc = [v for v in monitor.get_violations()
             if v.type == ViolationType.DTYPE_TRUNCATION]
    assert len(trunc) >= 1


def test_frozen_worker_seeds_detection():
    pytest.importorskip("xxhash")

    class StochasticDataset(Dataset):
        def __init__(self, n=100):
            self.x = torch.randn(n, 10)
            self.targets = [i % 3 for i in range(n)]
            self.transform = RandomNoiseTransform()

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.transform(self.x[i]), self.targets[i]

    def frozen_init(worker_id):
        torch.manual_seed(42)

    ds = StochasticDataset()
    loader = DataLoader(ds, batch_size=16, num_workers=2,
                        worker_init_fn=frozen_init)
    monitor = DLMonitor()
    monitor.guard(loader, role="train")
    frozen = [v for v in monitor.get_violations()
              if v.type == ViolationType.FROZEN_WORKER_SEEDS]
    assert len(frozen) >= 1


def test_frozen_worker_seeds_negative():
    pytest.importorskip("xxhash")

    class StochasticDataset(Dataset):
        def __init__(self, n=100):
            self.x = torch.randn(n, 10)
            self.targets = [i % 3 for i in range(n)]
            self.transform = RandomNoiseTransform()

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.transform(self.x[i]), self.targets[i]

    def proper_init(worker_id):
        torch.manual_seed(42 + worker_id)

    ds = StochasticDataset()
    loader = DataLoader(ds, batch_size=16, num_workers=2,
                        worker_init_fn=proper_init)
    monitor = DLMonitor()
    monitor.guard(loader, role="train")
    frozen = [v for v in monitor.get_violations()
              if v.type == ViolationType.FROZEN_WORKER_SEEDS]
    assert len(frozen) == 0


def test_partition_overlap_no_false_positive():
    monitor = DLMonitor()
    monitor.register_partition("train", list(range(0, 70)))
    monitor.register_partition("val", list(range(70, 100)))
    leaks = [v for v in monitor.check_partitions()
             if v.type == ViolationType.PARTITION_OVERLAP]
    assert len(leaks) == 0


def test_batch_distribution_no_false_positive():
    checker = BatchDistribution()
    checker.start_epoch(0)
    for i in range(20):
        checker.observe("class_0", 48.0 + 4.0 * (i % 3), i)
    checker.start_epoch(1)
    for i in range(20):
        vs = checker.observe("class_0", 48.0 + 4.0 * (i % 3), i)
        oob = [v for v in vs
               if v.type == ViolationType.BATCH_DISTRIBUTION_OOB]
        assert len(oob) == 0
